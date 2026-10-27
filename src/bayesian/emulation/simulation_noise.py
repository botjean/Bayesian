"""Opt-in known simulation covariance for PCA-GP training.

Independent GP priors are conditioned jointly because measured simulation
noise can correlate both design points and PC scores. Predictions target the
latent simulator mean: neither measured nor fitted residual noise is added a
second time at a new prediction point. The empirical PCA basis is held fixed.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import warnings

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.optimize import minimize
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process.kernels import Sum, WhiteKernel

from bayesian import data_IO


def configuration(settings):
    value = getattr(settings, 'settings', {}).get('simulation_noise')
    if value is None:
        return None
    required = {'covariance_file', 'residual_white_noise'}
    if not isinstance(value, dict) or not required <= set(value) or set(value) - required - {'joint_groups'}:
        raise ValueError('simulation_noise requires covariance_file and residual_white_noise')
    if 'joint_groups' in value and not isinstance(value['joint_groups'], bool):
        raise ValueError('simulation_noise.joint_groups must be boolean')
    if not isinstance(value['residual_white_noise'], bool):
        raise ValueError('simulation_noise.residual_white_noise must be boolean')
    path = Path(value['covariance_file'])
    if not path.is_absolute():
        raise ValueError('simulation_noise.covariance_file must be an absolute path')
    if value['residual_white_noise'] != ('noise' in settings.active_kernels):
        raise ValueError('simulation_noise.residual_white_noise must match the active noise kernel')
    result = {'covariance_file': str(path), 'residual_white_noise': value['residual_white_noise']}
    if 'joint_groups' in value:
        result['joint_groups'] = value['joint_groups']
    return result


def fingerprint(config):
    return {**config, 'sha256': hashlib.sha256(Path(config['covariance_file']).read_bytes()).hexdigest()}


def validate_cache(result, settings, check_file=False):
    config = configuration(settings)
    saved = result.get('simulation_noise')
    if (config is None) != (saved is None):
        raise ValueError('Cached simulation_noise differs; retrain before prediction')
    if config is not None:
        if {k: v for k, v in saved.items() if k != 'sha256'} != config or result.get('joint_pc_gp') is None:
            raise ValueError('Cached simulation_noise differs; retrain before prediction')
        if len(result['emulators']) != settings.n_pc:
            raise ValueError('Cached PC count differs; retrain before prediction')
        if check_file and fingerprint(config) != saved:
            raise ValueError('Simulation covariance file changed; retrain before prediction')


def validate_covariance(cov, size):
    cov = np.asarray(cov, dtype=float)
    if cov.shape != (size, size) or not np.isfinite(cov).all():
        raise ValueError('Simulation covariance has invalid shape or nonfinite entries')
    scale = max(float(np.max(abs(cov))), np.finfo(float).tiny)
    if np.max(abs(cov-cov.T)) > 1e-10*scale:
        raise ValueError('Simulation covariance must be symmetric')
    cov = (cov+cov.T)/2
    if np.linalg.eigvalsh(cov).min() < -1e-10*scale:
        raise ValueError('Simulation covariance must be positive semidefinite')
    return cov


def load_covariance(settings, analysis_settings):
    """Align an NPZ bank by design ID and observable/bin identity, not position.

    NPZ fields: design_ids (N,), observable_names/xmin/xmax (F,), and covariance
    (N,F,N,F) in unstandardized observable units. The bank may include held-out
    designs and unselected bins; these are selected using the training HDF5.
    """
    config = configuration(settings)
    raw = data_IO.read_dict_from_h5(analysis_settings.output_dir,
                                    analysis_settings.io.observables_filename, verbose=False)
    names = data_IO.sorted_observable_list_from_dict(raw, observable_filter=settings.base_settings.observable_filter)
    features = [(name,float(lo),float(hi)) for name in names
                for lo,hi in zip(raw['Data'][name]['xmin'],raw['Data'][name]['xmax'],strict=True)]
    return load_aligned_covariance(config['covariance_file'], raw['Design_indices'], features)


def load_aligned_covariance(path, ids, features):
    """Shared identity-based loader, also used for cross-group conditioning."""
    ids = np.asarray(ids)
    if len(set(features)) != len(features):
        raise ValueError('Selected simulation covariance features must be unique')
    with np.load(path, allow_pickle=False) as bank:
        source_ids = bank['design_ids']
        source_features = list(zip(bank['observable_names'].astype(str),bank['xmin'],bank['xmax'],strict=True))
        if source_ids.ndim != 1 or not np.issubdtype(source_ids.dtype,np.integer) or len(set(source_ids)) != len(source_ids):
            raise ValueError('Simulation covariance design IDs must be unique integers')
        if len(set(source_features)) != len(source_features):
            raise ValueError('Simulation covariance has duplicate feature identities')
        if len(set(ids)) != len(ids):
            raise ValueError('Training design IDs must be unique')
        try:
            di = np.array([list(source_ids).index(i) for i in ids])
            fi = np.array([source_features.index(f) for f in features])
        except ValueError as exc:
            raise ValueError('Simulation covariance is missing a training design or observable bin') from exc
        cov = bank['covariance']
        if cov.shape != (len(source_ids),len(source_features))*2:
            raise ValueError('Simulation covariance NPZ shape does not match its metadata')
        cov = cov[np.ix_(di,fi,di,fi)]
    n,f = len(di),len(fi)
    validate_covariance(cov.reshape(n*f,n*f),n*f)
    return cov


def project_covariance(cov, pca, scaler, n_pc):
    """Observable-unit covariance -> PC-major covariance, including whitening."""
    projection = pca.components_[:n_pc]/scaler.scale_[None,:]
    if pca.whiten:
        projection = projection/np.sqrt(pca.explained_variance_[:n_pc,None])
    projected = np.einsum('ab,ibjc,dc->aidj',projection,cov,projection,optimize=True)
    return projected.reshape(n_pc*cov.shape[0],n_pc*cov.shape[2])


def white_variance(kernel):
    if isinstance(kernel, WhiteKernel):
        return kernel.noise_level
    if isinstance(kernel, Sum):
        return white_variance(kernel.k1)+white_variance(kernel.k2)
    return 0.0


class JointPCGP:
    """Dense Gaussian conditioning with sklearn kernels and fixed known noise."""
    def __init__(self, kernels, jitter=1e-8, n_restarts=0, random_state=None):
        self.kernels = kernels
        self.jitter = jitter
        self.n_restarts = n_restarts
        self.random_state = random_state

    def _unpack(self, theta):
        offset = 0
        result = []
        for kernel in self.kernels:
            count = len(kernel.theta)
            result.append(kernel.clone_with_theta(theta[offset:offset+count]))
            offset += count
        return result

    def objective(self, theta, gradient=True):
        n,p = self.Y_train_.shape
        kernels = self._unpack(theta)
        cov = self.noise_.copy()
        cov.flat[::len(cov)+1] += self.jitter
        derivatives = []
        for i,kernel in enumerate(kernels):
            sl = slice(i*n,(i+1)*n)
            if gradient:
                block,grad = kernel(self.X_train_,eval_gradient=True)
                derivatives.append((sl,grad))
            else:
                block = kernel(self.X_train_)
            cov[sl,sl] += block
        try:
            L = cholesky(cov,lower=True,check_finite=False)
        except np.linalg.LinAlgError:
            return (np.inf,np.zeros_like(theta)) if gradient else np.inf
        target = self.Y_train_.T.ravel()
        alpha = cho_solve((L,True),target,check_finite=False)
        value = .5*target@alpha + np.log(np.diag(L)).sum() + .5*len(target)*np.log(2*np.pi)
        if not gradient:
            return float(value)
        matrix = cho_solve((L,True),np.eye(len(cov)),check_finite=False)-np.outer(alpha,alpha)
        grad = np.concatenate([.5*np.einsum('ij,ijk->k',matrix[sl,sl],d) for sl,d in derivatives])
        return float(value),grad

    def fit(self, x, y, noise, optimize=True, restart_thetas=None):
        self.X_train_ = np.array(x,dtype=float,copy=True)
        self.Y_train_ = np.array(y,dtype=float,copy=True)
        if (self.X_train_.ndim != 2 or self.Y_train_.ndim != 2
                or len(x) != len(y) or y.shape[1] != len(self.kernels)
                or not np.isfinite(x).all() or not np.isfinite(y).all()
                or not np.isfinite(self.jitter) or self.jitter < 0):
            raise ValueError('Invalid joint GP training inputs')
        self.noise_ = validate_covariance(noise,y.size)
        theta = np.concatenate([k.theta for k in self.kernels])
        bounds = np.vstack([k.bounds for k in self.kernels])
        self.optimization_ = []
        if optimize and len(theta):
            if not np.isfinite(bounds).all():
                raise ValueError('Joint GP restart bounds must be finite')
            rng = np.random.RandomState(self.random_state)
            if restart_thetas is None:
                restart_thetas = [rng.uniform(bounds[:,0],bounds[:,1]) for _ in range(self.n_restarts)]
            if len(restart_thetas) != self.n_restarts or any(
                np.shape(t) != theta.shape or not np.isfinite(t).all()
                or np.any(t < bounds[:, 0]) or np.any(t > bounds[:, 1]) for t in restart_thetas
            ):
                raise ValueError('Invalid explicit optimizer restarts')
            starts = [theta]+list(restart_thetas)
            candidates = []
            for initial in starts:
                fit = minimize(self.objective,initial,jac=True,bounds=bounds,method='L-BFGS-B')
                self.optimization_.append({'success':bool(fit.success),'message':str(fit.message),'objective':float(fit.fun)})
                candidates.append(fit)
            best = min(candidates,key=lambda fit:fit.fun)
            if not np.isfinite(best.fun):
                raise ValueError('No finite joint GP optimization result')
            theta = best.x
            if not best.success:
                warnings.warn(str(best.message),ConvergenceWarning)
            if np.any(np.isclose(theta,bounds[:,0],atol=1e-5,rtol=0) | np.isclose(theta,bounds[:,1],atol=1e-5,rtol=0)):
                warnings.warn('Joint PC GP has hyperparameters at configured bounds',ConvergenceWarning)
        self.kernels_ = self._unpack(theta)
        n,p = y.shape
        cov = self.noise_.copy()
        cov.flat[::len(cov)+1] += self.jitter
        for i,kernel in enumerate(self.kernels_):
            cov[i*n:(i+1)*n,i*n:(i+1)*n] += kernel(self.X_train_)
        self.L_ = cholesky(cov,lower=True)
        self.alpha_ = cho_solve((self.L_,True),y.T.ravel())
        self.log_marginal_likelihood_value_ = -self.objective(theta,gradient=False)
        return self

    def predict(self, points, observation_noise=None, observation_cross_noise=None):
        """Return pointwise full PC covariance. Optional noise is for held-out QA only.

        observation_noise is (P*M,P*M); cross noise is (P*M,P*N), PC-major.
        This allows validation against noisy targets sharing the training pp
        reference, without pretending that their MC noise is independent.
        """
        points = np.asarray(points,dtype=float)
        m,n,p = len(points),len(self.X_train_),len(self.kernels_)
        cross = np.zeros((p*m,p*n))
        prior = np.zeros((m,p,p))
        for i,kernel in enumerate(self.kernels_):
            cross[i*m:(i+1)*m,i*n:(i+1)*n] = kernel(points,self.X_train_)
            prior[:,i,i] = kernel.diag(points)-white_variance(kernel)
        if observation_noise is not None:
            if observation_cross_noise is None or observation_noise.shape != (p*m,p*m) or observation_cross_noise.shape != cross.shape:
                raise ValueError('Observed-target prediction needs aligned test and cross noise')
            cross += observation_cross_noise
            for i in range(p):
                for j in range(p):
                    prior[:,i,j] += np.diag(observation_noise[i*m:(i+1)*m,j*m:(j+1)*m])
                prior[:,i,i] += white_variance(self.kernels_[i])
        elif observation_cross_noise is not None:
            raise ValueError('Cross noise without test noise')
        mean = (cross@self.alpha_).reshape(p,m).T
        v = solve_triangular(self.L_,cross.T,lower=True,check_finite=False).reshape(p*n,p,m)
        cov = prior-np.einsum('aim,ajm->mij',v,v,optimize=True)
        cov = (cov+cov.swapaxes(1,2))/2
        tolerance = 1e-8*max(1.0,float(abs(prior).max()))
        if np.linalg.eigvalsh(cov).min() < -tolerance:
            raise ValueError('Joint GP predictive covariance is not PSD')
        return mean,cov


class PCView:
    """Small compatibility view for native single-PC QA; inference uses joint GP."""
    def __init__(self, joint, index):
        self.joint = joint
        self.index = index
        self.kernel_ = joint.kernels_[index]
        self.X_train_ = joint.X_train_
        self.log_marginal_likelihood_value_ = joint.log_marginal_likelihood_value_

    def predict(self, x, return_std=False, return_cov=False):
        if return_cov:
            raise ValueError('Use joint PC covariance prediction; this view supplies marginal std only')
        mean,cov = self.joint.predict(x)
        if return_std:
            return mean[:,self.index],np.sqrt(np.maximum(cov[:,self.index,self.index],0))
        return mean[:,self.index]

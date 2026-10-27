"""Mean-aware REML and prediction with measured correlated simulation noise.

These explicit opt-in classes marginalize unknown constant means. Ordinary
pipeline defaults and existing trained caches are not changed by importing them.
"""
import numpy as np
from scipy.linalg import cholesky, cho_solve, helmert, solve_triangular

from bayesian.emulation.simulation_noise import JointPCGP
from bayesian.emulation.omitted_coordinates import omitted_maps


class MeanAware:
    def __init__(self, gp, truth, projection, transfer, center, residual,
                 omitted_projection, omitted_transfer, noise):
        self.gp = gp
        self.projection = projection
        self.transfer = transfer
        self.center = center
        self.residual = residual
        self.omitted_projection = omitted_projection
        self.omitted_transfer = omitted_transfer
        n, f = truth.shape
        p, o = len(projection), len(omitted_projection)
        if noise.shape != (n, f, n, f) or p+o != f:
            raise ValueError('Incomplete/alignment-mismatched mean model')
        self.n, self.f, self.p, self.o = n, f, p, o
        forward = np.vstack([projection, omitted_projection])
        self.inverse = np.vstack([transfer, omitted_transfer])
        np.testing.assert_allclose(forward @ self.inverse.T, np.eye(f), atol=1e-10)
        np.testing.assert_allclose(self.inverse.T @ forward, np.eye(f), atol=1e-10)
        self.noise = noise
        self.omitted_covariance = omitted_projection @ residual @ omitted_projection.T
        np.testing.assert_allclose(projection @ residual, 0, atol=1e-10)
        rr = np.einsum('pa,iajb,qb->piqj', projection, noise, projection,
                       optimize=True).reshape(p*n, p*n)
        ro = np.einsum('pa,iajb,qb->piq', projection, noise/n,
                       omitted_projection, optimize=True).reshape(p*n, o)
        oo = omitted_projection @ noise.mean(axis=(0, 2)) @ omitted_projection.T
        self.S = np.block([[rr, ro], [ro.T, oo+self.omitted_covariance/n]])
        self.S[:p*n, :p*n].flat[::p*n+1] += gp.jitter
        for i, kernel in enumerate(gp.kernels_):
            sl = slice(i*n, (i+1)*n)
            self.S[sl, sl] += kernel(gp.X_train_)
        self.L = cholesky((self.S+self.S.T)/2, lower=True)
        self.H = np.zeros((p*n+o, f))
        for i in range(p):
            self.H[i*n:(i+1)*n, i] = 1
        self.H[p*n:, p:] = np.eye(o)
        centered = truth-center
        self.y = np.r_[(centered @ projection.T).T.ravel(),
                       centered.mean(axis=0) @ omitted_projection.T]
        np.testing.assert_allclose(self.y[:p*n], gp.Y_train_.T.ravel(), atol=1e-10)
        self.SinvH = cho_solve((self.L, True), self.H)
        normal = self.H.T @ self.SinvH
        self.mean_L = cholesky((normal+normal.T)/2, lower=True)
        self.beta = cho_solve((self.mean_L, True),
                             self.H.T @ cho_solve((self.L, True), self.y))
        self.alpha = cho_solve((self.L, True), self.y-self.H @ self.beta)
        self.mean_covariance = cho_solve((self.mean_L, True), np.eye(f))

    def predict(self, points, target_noise=None):
        """Latent predictions, or noisy heldouts with exact test/train MC noise."""
        points = np.asarray(points)
        m, n, p, f = len(points), self.n, self.p, self.f
        kernel_cross = np.array([k(points, self.gp.X_train_)
                                 for k in self.gp.kernels_])
        cross = np.zeros((m, f, p*n+self.o))
        cross[:, :, :p*n] = np.einsum('pa,pmn->mapn', self.transfer,
                                     kernel_cross).reshape(m, f, p*n)
        prior = np.einsum('pa,mp,pb->mab', self.transfer,
                         np.column_stack([k.diag(points) for k in self.gp.kernels_]),
                         self.transfer)+self.residual
        if target_noise is not None:
            if target_noise.shape != (n+m, f, n+m, f):
                raise ValueError('Invalid test/train noise alignment')
            train_test = target_noise[n:, :, :n, :]
            cross[:, :, :p*n] += np.einsum('iajb,pb->iapj', train_test,
                                          self.projection).reshape(m, f, p*n)
            cross[:, :, p*n:] += np.einsum('iab,pb->iap', train_test.mean(axis=2),
                                          self.omitted_projection)
            prior += np.stack([target_noise[n+i, :, n+i, :] for i in range(m)])
        flat = cross.reshape(m*f, -1)
        mean = self.center+self.beta @ self.inverse+(flat @ self.alpha).reshape(m, f)
        solved = solve_triangular(self.L, flat.T, lower=True,
                                  check_finite=False).reshape(-1, m, f)
        known = prior-np.einsum('ima,imb->mab', solved, solved, optimize=True)
        correction = self.inverse.T[None, :, :]-cross @ self.SinvH
        mean_part = np.einsum('mai,ij,mbj->mab', correction,
                              self.mean_covariance, correction, optimize=True)
        cov = known+mean_part
        cov = (cov+cov.swapaxes(1, 2))/2
        if np.linalg.eigvalsh(cov).min() < -1e-9*max(float(abs(prior).max()), 1):
            raise ValueError('Mean-aware predictive covariance is not PSD')
        return dict(central_value=mean, cov=cov,
                    fitted_mean_covariance=mean_part)


class RestrictedMeanGP(JointPCGP):
    """Integrate flat constant PC means using orthonormal design contrasts.

This is the same universal-mean model used in prediction. The omitted-score
average has a free mean of its own, so integrating that mean removes it from
the kernel objective exactly, not approximately.
"""
    def fit(self, x, y, noise, **kwargs):
        n, p = y.shape
        self.contrasts_ = helmert(n, full=False)
        self.contrast_target_ = (self.contrasts_ @ y).T.ravel()
        self.contrast_noise_ = np.einsum('hi,piqj,kj->phqk', self.contrasts_,
            noise.reshape(p, n, p, n), self.contrasts_, optimize=True).reshape(p*(n-1), -1)
        super().fit(x, y, noise, **kwargs)
        h = np.kron(np.eye(p), np.ones((n, 1)))
        self.mean_SinvH_ = cho_solve((self.L_, True), h)
        normal = h.T @ self.mean_SinvH_
        self.mean_cov_ = cho_solve((cholesky((normal+normal.T)/2, lower=True), True), np.eye(p))
        self.mean_beta_ = self.mean_cov_ @ h.T @ self.alpha_
        self.mean_alpha_ = cho_solve((self.L_, True), y.T.ravel()-h @ self.mean_beta_)
        return self

    def objective(self, theta, gradient=True):
        n, p = self.Y_train_.shape
        q = self.contrasts_
        covariance = self.contrast_noise_.copy()
        covariance.flat[::len(covariance)+1] += self.jitter
        derivatives = []
        for i, kernel in enumerate(self._unpack(theta)):
            sl = slice(i*(n-1), (i+1)*(n-1))
            if gradient:
                block, grad = kernel(self.X_train_, eval_gradient=True)
                derivatives.append((sl, np.einsum('hi,ijt,kj->hkt', q, grad, q, optimize=True)))
            else:
                block = kernel(self.X_train_)
            covariance[sl, sl] += q @ block @ q.T
        try:
            factor = cholesky((covariance+covariance.T)/2, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            return (np.inf, np.zeros_like(theta)) if gradient else np.inf
        alpha = cho_solve((factor, True), self.contrast_target_, check_finite=False)
        value = (.5*self.contrast_target_ @ alpha+np.log(np.diag(factor)).sum()
                 +.5*len(alpha)*np.log(2*np.pi))
        if not gradient:
            return float(value)
        weight = cho_solve((factor, True), np.eye(len(alpha)), check_finite=False)-np.outer(alpha, alpha)
        derivative = np.concatenate([.5*np.einsum('ij,ijt->t', weight[sl, sl], grad)
                                     for sl, grad in derivatives])
        return float(value), derivative

    def predict(self, points, observation_noise=None, observation_cross_noise=None):
        """Native PC views must use the refitted unknown-mean posterior too."""
        points = np.asarray(points)
        n, p, m = len(self.X_train_), len(self.kernels_), len(points)
        cross = np.zeros((p*m, p*n))
        prior = np.zeros((m, p, p))
        for i, kernel in enumerate(self.kernels_):
            cross[i*m:(i+1)*m, i*n:(i+1)*n] = kernel(points, self.X_train_)
            prior[:, i, i] = kernel.diag(points)
        if observation_noise is not None:
            if observation_cross_noise is None or observation_noise.shape != (p*m, p*m):
                raise ValueError('Aligned test/train noise required')
            if observation_cross_noise.shape != cross.shape:
                raise ValueError('Wrong cross-noise shape')
            cross += observation_cross_noise
            for i in range(p):
                for j in range(p):
                    prior[:, i, j] += np.diag(observation_noise[i*m:(i+1)*m, j*m:(j+1)*m])
        elif observation_cross_noise is not None:
            raise ValueError('Cross noise without test noise')
        mean = self.mean_beta_[None, :]+(cross @ self.mean_alpha_).reshape(p, m).T
        solved = solve_triangular(self.L_, cross.T, lower=True, check_finite=False).reshape(p*n, p, m)
        covariance = prior-np.einsum('aim,ajm->mij', solved, solved, optimize=True)
        correction = np.eye(p)[None, :, :]-(cross @ self.mean_SinvH_).reshape(p, m, p).transpose(1, 0, 2)
        covariance += np.einsum('mai,ij,mbj->mab', correction, self.mean_cov_, correction, optimize=True)
        return mean, (covariance+covariance.swapaxes(1, 2))/2


class FastMeanAware(MeanAware):
    """Algebraically identical latent prediction in retained-PC space."""
    def predict(self, points, target_noise=None):
        if target_noise is not None:
            return super().predict(points, target_noise)
        points = np.asarray(points)
        kernels = np.array([k(points, self.gp.X_train_) for k in self.gp.kernels_])
        mean_pc = np.einsum('pmn,pn->mp', kernels, self.alpha[:self.p*self.n].reshape(self.p, self.n))
        mean = self.center+self.beta @ self.inverse+mean_pc @ self.transfer
        cross_pc = np.zeros((len(points), self.p, len(self.y)))
        for i in range(self.p):
            cross_pc[:, i, i*self.n:(i+1)*self.n] = kernels[i]
        solved = solve_triangular(self.L, cross_pc.reshape(-1, len(self.y)).T,
                                  lower=True, check_finite=False).reshape(len(self.y), len(points), self.p)
        covariance_pc = -np.einsum('ima,imb->mab', solved, solved, optimize=True)
        for i, kernel in enumerate(self.gp.kernels_):
            covariance_pc[:, i, i] += kernel.diag(points)
        covariance = np.einsum('pa,mpq,qb->mab', self.transfer, covariance_pc, self.transfer, optimize=True)+self.residual
        cross_h = np.einsum('pmn,pnf->mpf', kernels,
            self.SinvH[:self.p*self.n].reshape(self.p, self.n, self.f), optimize=True)
        correction = self.inverse.T[None, :, :]-np.einsum('pa,mpf->maf', self.transfer, cross_h, optimize=True)
        mean_part = np.einsum('mai,ij,mbj->mab', correction, self.mean_covariance, correction, optimize=True)
        covariance += mean_part
        covariance = (covariance+covariance.swapaxes(1, 2))/2
        if np.linalg.eigvalsh(covariance).min() < -1e-9:
            raise ValueError('Mean-aware covariance failed PSD')
        return dict(central_value=mean, cov=covariance, fitted_mean_covariance=mean_part)

"""Known noise is propagated, aligned, and conditioned, never added twice."""
from types import SimpleNamespace
import pickle

import numpy as np
import pytest
from scipy.linalg import block_diag
from scipy.optimize._numdiff import approx_derivative
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler

from bayesian.emulation import simulation_noise as sn
from bayesian.emulation import sk_learn
from tests.test_emulation_signal_amplitude import configuration


def toy():
    x = np.linspace(0,1,7)[:,None]
    y = np.column_stack((np.sin(3*x[:,0]),np.cos(2*x[:,0])))
    kernels = [ConstantKernel(1.3)*RBF(.4),ConstantKernel(.7)*RBF(.9)]
    rng = np.random.default_rng(13)
    factor = rng.normal(size=(14,8))*.07
    noise = factor@factor.T+np.eye(14)*.02
    return x,y,kernels,noise


def test_diagonal_noise_equals_sklearn_alpha():
    x,y,kernels,_ = toy()
    variances = np.linspace(.01,.1,14)
    joint = sn.JointPCGP(kernels).fit(x,y,np.diag(variances),optimize=False)
    points = np.linspace(-.1,1.1,9)[:,None]
    mean,cov = joint.predict(points)
    for pc in range(2):
        ref = GaussianProcessRegressor(kernel=kernels[pc],alpha=variances[pc*7:(pc+1)*7]+1e-8,optimizer=None).fit(x,y[:,pc])
        mu,sd = ref.predict(points,return_std=True)
        np.testing.assert_allclose(mean[:,pc],mu,atol=1e-12)
        np.testing.assert_allclose(cov[:,pc,pc],sd**2,atol=1e-12)
    np.testing.assert_allclose(cov[:,0,1],0,atol=1e-12)


@pytest.mark.parametrize('residual',[False,True])
def test_full_noise_against_direct_conditioning(residual):
    x,y,kernels,noise = toy()
    if residual:
        kernels = [k+WhiteKernel(.12) for k in kernels]
    joint = sn.JointPCGP(kernels).fit(x,y,noise,optimize=False)
    points = np.array([[.17],[.7]])
    cross = block_diag(*[k(points,x) for k in kernels])
    # Latent prior excludes fitted white noise as well as known simulation noise.
    prior = block_diag(*[k(points,points) for k in kernels])
    train = block_diag(*[k(x) for k in kernels])+noise+np.eye(14)*1e-8
    direct_mean = cross@np.linalg.solve(train,y.T.ravel())
    direct_cov = prior-cross@np.linalg.solve(train,cross.T)
    mean,cov = joint.predict(points)
    np.testing.assert_allclose(mean,direct_mean.reshape(2,2).T,atol=1e-12)
    for i in range(2):
        np.testing.assert_allclose(cov[i],direct_cov[np.ix_([i,i+2],[i,i+2])],atol=1e-12)
    assert np.max(abs(cov[:,0,1])) > 1e-6
    mu,co = pickle.loads(pickle.dumps(joint)).predict(points)
    np.testing.assert_array_equal(mu,mean)
    np.testing.assert_array_equal(co,cov)
    for i in range(2):
        mm,cc = joint.predict(points[i:i+1])
        np.testing.assert_allclose(mm[0],mean[i],atol=1e-12)
        np.testing.assert_allclose(cc[0],cov[i],atol=1e-12)


def test_objective_gradient_matches_finite_difference():
    x,y,kernels,noise = toy()
    joint = sn.JointPCGP(kernels).fit(x,y,noise,optimize=False)
    theta = np.concatenate([k.theta for k in kernels])
    _,grad = joint.objective(theta)
    finite = approx_derivative(lambda t:joint.objective(t,gradient=False),theta).ravel()
    np.testing.assert_allclose(grad,finite,atol=1e-7,rtol=1e-5)


def test_noisy_validation_conditions_on_shared_noise():
    x,y,kernels,noise = toy()
    joint = sn.JointPCGP(kernels).fit(x,y,noise,optimize=False)
    points = np.array([[.3]])
    crossnoise = noise[[2,9]]*.2
    testnoise = noise[np.ix_([2,9],[2,9])]*.04+np.eye(2)*.1
    mu,cov = joint.predict(points,testnoise,crossnoise)
    cross = block_diag(*[k(points,x) for k in kernels])+crossnoise
    prior = block_diag(*[k(points) for k in kernels])+testnoise
    train = block_diag(*[k(x) for k in kernels])+noise+np.eye(14)*1e-8
    np.testing.assert_allclose(mu[0],cross@np.linalg.solve(train,y.T.ravel()))
    np.testing.assert_allclose(cov[0],prior-cross@np.linalg.solve(train,cross.T))


@pytest.mark.parametrize('whiten',[False,True])
def test_projection_matches_explicit_linear_transform(whiten):
    rng = np.random.default_rng(22)
    y = rng.normal(size=(6,3))*[.1,2,10]
    scaler = StandardScaler().fit(y)
    pca = PCA(whiten=whiten).fit(scaler.transform(y))
    a = rng.normal(size=(18,18))
    cov = a@a.T
    proj = pca.components_[:2]/scaler.scale_
    if whiten:
        proj = proj/np.sqrt(pca.explained_variance_[:2,None])
    operator = np.kron(np.eye(6),proj)
    ix = np.arange(12).reshape(6,2).T.ravel()
    expected = (operator@cov@operator.T)[np.ix_(ix,ix)]
    actual = sn.project_covariance(cov.reshape(6,3,6,3),pca,scaler,2)
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-10)


@pytest.mark.parametrize('cov',[np.eye(2)*np.nan, np.array([[1,2],[0,1]]),np.diag([1,-.1]),np.ones((2,3))])
def test_invalid_covariance_rejected(cov):
    with pytest.raises(ValueError):
        sn.validate_covariance(cov,2)


def test_loader_aligns_ids_bins_and_rejects_missing(tmp_path,monkeypatch):
    path = tmp_path/'cov.npz'
    cfg = configuration()
    cfg['simulation_noise'] = {'covariance_file':str(path),'residual_white_noise':True}
    settings = sk_learn.SKLearnEmulatorSettings.from_config(cfg)
    raw = {'Design_indices':np.array([30,10]),'Data':{'toy':{'xmin':np.array([1.,0.]),'xmax':np.array([2.,1.])}}}
    monkeypatch.setattr(sn.data_IO,'read_dict_from_h5',lambda *a,**k:raw)
    monkeypatch.setattr(sn.data_IO,'sorted_observable_list_from_dict',lambda *a,**k:['toy'])
    cov = np.diag(np.arange(1,7,dtype=float)).reshape(3,2,3,2)
    np.savez(path,design_ids=[10,20,30],observable_names=['toy','toy'],xmin=[0.,1.],xmax=[1.,2.],covariance=cov)
    analysis = SimpleNamespace(output_dir=tmp_path,io=SimpleNamespace(observables_filename='test'))
    actual = sn.load_covariance(settings,analysis).reshape(4,4)
    np.testing.assert_array_equal(np.diag(actual),[6,5,2,1])
    raw['Design_indices'] = np.array([99])
    with pytest.raises(ValueError,match='missing'):
        sn.load_covariance(settings,analysis)


def test_cache_option_and_source_changes_rejected(tmp_path):
    path = tmp_path/'cov.npz'
    np.savez(path,example=[1])
    cfg = configuration()
    legacy = sk_learn.SKLearnEmulatorSettings.from_config(cfg)
    cfg = {**cfg,'simulation_noise':{'covariance_file':str(path),'residual_white_noise':True}}
    settings = sk_learn.SKLearnEmulatorSettings.from_config(cfg)
    with pytest.raises(ValueError,match='retrain'):
        sn.validate_cache({},settings)
    result = {'simulation_noise':sn.fingerprint(cfg['simulation_noise']),'joint_pc_gp':True,'emulators':[0,1]}
    sn.validate_cache(result,settings,check_file=True)
    with pytest.raises(ValueError,match='retrain'):
        sn.validate_cache(result,legacy)
    np.savez(path,example=[2])
    with pytest.raises(ValueError,match='changed'):
        sn.validate_cache(result,settings,check_file=True)


@pytest.mark.filterwarnings('ignore::sklearn.exceptions.ConvergenceWarning')
def test_opt_in_native_fit_predict(tmp_path,monkeypatch):
    x,y,_,noise = toy()
    monkeypatch.setattr(sk_learn.data_IO,'predictions_matrix_from_h5',lambda **kwargs:y)
    monkeypatch.setattr(sk_learn.data_IO,'design_array_from_h5',lambda *args,**kwargs:x)
    # Feature covariance uses design-major ordering; projection handles PC ordering.
    feature_noise = np.eye(y.size)*.04
    monkeypatch.setattr(sn,'load_covariance',lambda *args:feature_noise.reshape(7,2,7,2))
    path = tmp_path/'cov.npz'
    np.savez(path,covariance=feature_noise)
    cfg = configuration()
    cfg['simulation_noise'] = {'covariance_file':str(path),'residual_white_noise':True}
    settings = sk_learn.SKLearnEmulatorSettings.from_config(cfg)
    analysis = SimpleNamespace(output_dir=tmp_path,io=SimpleNamespace(observables_filename='unused'),parameterization='toy',
        raw_analysis_config={'parameterization':{'toy':{'min':[0.],'max':[1.]}}})
    result = sk_learn.fit_emulator(settings,analysis)
    pred = sk_learn.predict(x[:3],result,settings)
    assert pred['central_value'].shape == (3,2)
    assert np.linalg.eigvalsh(pred['cov']).min() >= 0
    assert result['joint_pc_gp'] is not None

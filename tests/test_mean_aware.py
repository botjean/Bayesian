import numpy as np
import pytest
from scipy.linalg import block_diag
from scipy.optimize._numdiff import approx_derivative
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from bayesian.emulation.mean_aware import MeanAware, RestrictedMeanGP, FastMeanAware


def toy(y_shift=None):
    rng = np.random.default_rng(16)
    x = np.linspace(0, 1, 5)[:, None]
    y = rng.normal(0, .2, (5, 2))
    if y_shift is not None:
        y += y_shift
    a = rng.normal(size=(10, 10))*.02
    noise = a @ a.T+np.eye(10)*.001
    kernels = [ConstantKernel(.05, (.001, 2))*Matern(.3, (.01, 10), nu=1.5),
               ConstantKernel(.08, (.001, 2))*Matern(.5, (.01, 10), nu=1.5)]
    return RestrictedMeanGP(kernels, jitter=1e-8).fit(x, y, noise, optimize=False)


def test_objective_gradient_by_finite_differences():
    gp = toy()
    theta = np.concatenate([k.theta for k in gp.kernels_])
    value, gradient = gp.objective(theta)
    actual = approx_derivative(lambda z: gp.objective(z, gradient=False), theta).ravel()
    np.testing.assert_allclose(gradient, actual, rtol=1e-6, atol=1e-7)
    assert np.isfinite(value)


def test_REML_equals_explicit_mean_integral():
    gp = toy()
    n, p = gp.Y_train_.shape
    covariance = gp.noise_+block_diag(*[k(gp.X_train_) for k in gp.kernels_])+np.eye(n*p)*gp.jitter
    h = np.kron(np.eye(p), np.ones((n, 1)))
    inverse = np.linalg.inv(covariance)
    normal = h.T @ inverse @ h
    precision = inverse-inverse @ h @ np.linalg.inv(normal) @ h.T @ inverse
    target = gp.Y_train_.T.ravel()
    direct = (.5*target @ precision @ target+.5*np.linalg.slogdet(covariance)[1]
              +.5*np.linalg.slogdet(normal)[1]-.5*np.linalg.slogdet(h.T @ h)[1]
              +.5*(n*p-p)*np.log(2*np.pi))
    theta = np.concatenate([k.theta for k in gp.kernels_])
    np.testing.assert_allclose(gp.objective(theta, gradient=False), direct, atol=1e-11)


def test_refit_mean_is_translation_invariant():
    a, b = toy(), toy(np.array([3., -2.]))
    theta = np.concatenate([k.theta for k in a.kernels_])
    np.testing.assert_allclose(a.objective(theta)[0], b.objective(theta)[0], atol=1e-11)
    am, ac = a.predict([[.25], [.75]])
    bm, bc = b.predict([[.25], [.75]])
    np.testing.assert_allclose(bm, am+np.array([3., -2.]), atol=1e-11)
    np.testing.assert_allclose(bc, ac, atol=1e-11)


def test_fast_prediction_is_exact_and_native_PC_views_agree():
    gp = toy()
    n = len(gp.X_train_)
    truth = np.column_stack([gp.Y_train_, np.zeros(n)])
    projection = np.array([[1., 0, 0], [0, 1., 0]])
    omitted = np.array([[0., 0, 1.]])
    noise = np.zeros((n, 3, n, 3))
    for i in range(2):
        for j in range(2):
            noise[:, i, :, j] = gp.noise_[i*n:(i+1)*n, j*n:(j+1)*n]
    noise[:, 2, :, 2] = .05+np.eye(n)*.01
    args = (gp, truth, projection, projection, np.zeros(3), np.diag([0., 0, .02]), omitted, omitted, noise)
    full, fast = MeanAware(*args), FastMeanAware(*args)
    points = [[.2], [.5], [.9]]
    a, b = full.predict(points), fast.predict(points)
    for field in a:
        np.testing.assert_allclose(a[field], b[field], atol=1e-11, rtol=1e-10)
    mean, covariance = gp.predict(points)
    np.testing.assert_allclose(mean, b['central_value'][:, :2], atol=1e-11)
    np.testing.assert_allclose(covariance, b['cov'][:, :2, :2], atol=1e-11)


@pytest.mark.filterwarnings('ignore:Joint PC GP has hyperparameters at configured bounds:sklearn.exceptions.ConvergenceWarning')
def test_optimizer_really_updates_kernels_with_mean_likelihood():
    gp = toy()
    theta = np.concatenate([k.theta for k in gp.kernels_])
    initial = gp.objective(theta, gradient=False)
    gp.fit(gp.X_train_, gp.Y_train_, gp.noise_, optimize=True)
    assert -gp.log_marginal_likelihood_value_ <= initial+1e-9
    assert np.max(abs(np.concatenate([k.theta for k in gp.kernels_])-theta)) > .01

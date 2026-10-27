from types import SimpleNamespace as NS

import numpy as np
import pytest
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

from bayesian.emulation.mean_aware import MeanAware
from bayesian.emulation.omitted_coordinates import omitted_maps


def fixture(whiten=False, complete=False):
    rng = np.random.default_rng(817)
    n, f, p = 4, 7, 2
    q = np.linalg.qr(rng.normal(size=(f, f)))[0].T
    variances = np.linspace(1.8, .2, f)
    scale = np.linspace(.6, 1.4, f)
    rows = f if complete else n
    pca = NS(components_=q[:rows], explained_variance_=variances[:rows], whiten=whiten)
    config = NS(sort_observables_in_matrix=NS(shape=(n, f)), emulation_settings={'g': NS(n_pc=p)})
    models = {'g': {'PCA': {'pca': pca, 'scaler': NS(scale_=scale)}}}
    factors = np.sqrt(variances[:p]) if whiten else np.ones(p)
    projection = q[:p] / scale / factors[:, None]
    transfer = q[:p] * scale * factors[:, None]
    x = np.linspace(0, 1, n)[:, None]
    kernels = [ConstantKernel(.07 * (i+1)) * RBF(.35 + .1*i) for i in range(p)]
    truth = rng.normal(size=(n, f))
    center = truth.mean(0)
    residual = (q[p:rows].T * variances[p:rows]) @ q[p:rows]
    residual = residual * np.outer(scale, scale)
    a = rng.normal(size=(n*f, n*f))
    noise = (a @ a.T / 800 + np.eye(n*f)*.01).reshape(n, f, n, f)
    gp = NS(X_train_=x, Y_train_=(truth-center) @ projection.T, kernels_=kernels, jitter=1e-8)
    u, b = omitted_maps(config, models, {'g': np.arange(f)})
    model = MeanAware(gp, truth, projection, transfer, center, residual, u, b, noise)
    return model, truth, config, models, q, scale


@pytest.mark.parametrize('whiten', [False, True])
def test_complete_maps_bitwise_unchanged(whiten):
    _, _, config, models, _, _ = fixture(whiten, True)
    indices = {'g': np.arange(7)}
    actual = omitted_maps(config, models, indices)
    pca = models['g']['PCA']['pca']
    scale = models['g']['PCA']['scaler'].scale_
    rows = pca.components_[2:]
    factors = np.sqrt(pca.explained_variance_[2:]) if whiten else np.ones(len(rows))
    expected = (rows / scale / factors[:, None], rows * scale * factors[:, None])
    for x, y in zip(actual, expected):
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize('whiten', [False, True])
def test_incomplete_coordinates_match_independent_raw_feature_oracle(whiten):
    model, truth, _, _, _, _ = fixture(whiten)
    n, f, p = model.n, model.f, model.p
    # Independently build raw-feature covariance and compress its observations.
    full = model.noise.reshape(n*f, n*f).copy() + np.kron(np.eye(n), model.residual)
    for kernel, direction in zip(model.gp.kernels_, model.transfer):
        full += np.kron(kernel(model.gp.X_train_), np.outer(direction, direction))
    full += np.kron(np.eye(n)*model.gp.jitter, model.transfer.T @ model.transfer)
    transform = np.zeros((p*n+model.o, n*f))
    for i in range(p):
        for j in range(n):
            transform[i*n+j, j*f:(j+1)*f] = model.projection[i]
    transform[p*n:] = np.tile(model.omitted_projection/n, (1, n))
    S = transform @ full @ transform.T
    np.testing.assert_allclose(S, model.S, atol=2e-14)
    H = transform @ np.tile(np.eye(f), (n, 1))
    y = transform @ (truth-model.center).ravel()
    inv = np.linalg.inv(S)
    beta_cov = np.linalg.inv(H.T @ inv @ H)
    beta = beta_cov @ H.T @ inv @ y
    point = np.array([[.27]])
    cross_raw = sum(np.kron(kernel(point, model.gp.X_train_), np.outer(t, t))
                    for kernel, t in zip(model.gp.kernels_, model.transfer))
    cross = cross_raw @ transform.T
    prior = model.residual + sum(kernel.diag(point)[0]*np.outer(t, t)
                                for kernel, t in zip(model.gp.kernels_, model.transfer))
    correction = np.eye(f)-cross @ inv @ H
    expected_mean = model.center+beta+cross @ inv @ (y-H @ beta)
    expected_cov = prior-cross @ inv @ cross.T+correction @ beta_cov @ correction.T
    pred = model.predict(point)
    np.testing.assert_allclose(pred['central_value'][0], expected_mean, atol=2e-11)
    np.testing.assert_allclose(pred['cov'][0], expected_cov, atol=2e-11)
    # Test shared test/train MC noise with an independently augmented raw block.
    rng = np.random.default_rng(915)
    c = rng.normal(size=(f, n*f)) * .0001
    test = np.eye(f)*.03
    joint = np.zeros((n+1, f, n+1, f))
    joint[:n, :, :n, :] = model.noise
    joint[n, :, :n, :] = c.reshape(f, n, f)
    joint[:n, :, n, :] = c.T.reshape(n, f, f)
    joint[n, :, n, :] = test
    observed_cross = (cross_raw+c) @ transform.T
    correction = np.eye(f)-observed_cross @ inv @ H
    expected_cov = prior+test-observed_cross @ inv @ observed_cross.T+correction @ beta_cov @ correction.T
    observed = model.predict(point, joint)
    np.testing.assert_allclose(observed['cov'][0], expected_cov, atol=2e-11)
    np.testing.assert_allclose(observed['central_value'][0],
        model.center+beta+observed_cross @ inv @ (y-H @ beta), atol=2e-11)


@pytest.mark.parametrize('whiten', [False, True])
def test_omitted_rotation_invariant_including_missing_svd_directions(whiten):
    model, truth, _, _, q, scale = fixture(whiten)
    # The independent complement uses the full orthogonal matrix from the toy.
    u, b = q[model.p:]/scale, q[model.p:]*scale
    oracle = MeanAware(model.gp, truth, model.projection, model.transfer,
        model.center, model.residual, u, b, model.noise)
    for field, value in model.predict([[.2], [.8]]).items():
        np.testing.assert_allclose(value, oracle.predict([[.2], [.8]])[field], atol=2e-11)
    # Missing native rows have zero R, but their mean noise must not disappear.
    direction = q[-1]/scale
    assert np.linalg.norm(model.residual @ (q[-1]/scale)) < 1e-12
    assert direction @ model.predict([[.4]])['cov'][0] @ direction > 1e-5


def test_noncontiguous_multigroup_mapping():
    _, _, cfg, models, _, _ = fixture()
    config = NS(sort_observables_in_matrix=NS(shape=(4, 14)),
        emulation_settings={'a': cfg.emulation_settings['g'], 'b': cfg.emulation_settings['g']})
    mapping = {'a': np.arange(0, 14, 2), 'b': np.arange(1, 14, 2)}
    u, b = omitted_maps(config, {'a': models['g'], 'b': models['g']}, mapping)
    assert u.shape == b.shape == (10, 14)
    np.testing.assert_allclose(u @ b.T, np.eye(10), atol=1e-12)

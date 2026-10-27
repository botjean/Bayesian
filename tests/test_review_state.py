from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from bayesian.emulation.mean_aware import FastMeanAware
from bayesian.emulation.review_state import load_prediction_state, restore_kernel, save_prediction_state


def fixture():
    n = 4
    x = np.linspace(0, 1, n)[:, None]
    truth = np.column_stack([np.sin(x[:, 0]), np.cos(x[:, 0]), np.zeros(n)])
    projection = np.eye(3)[:2]
    omitted = np.eye(3)[2:]
    kernels = [ConstantKernel(.1)*Matern(.3, nu=1.5) for _ in range(2)]
    gp = SimpleNamespace(X_train_=x, Y_train_=truth[:, :2], kernels_=kernels, jitter=1e-8)
    noise = (np.eye(n*3)*.02 + np.ones((n*3, n*3))*.001).reshape(n, 3, n, 3)
    model = FastMeanAware(gp, truth, projection, projection, np.zeros(3),
                          np.diag([0, 0, .01]), omitted, omitted, noise)
    features = [('toy', i, i+1) for i in range(3)]
    return model, features, noise


def test_roundtrip_predictions_and_paired_noise(tmp_path):
    model, features, noise = fixture()
    path = tmp_path/'state.npz'
    save_prediction_state(path, model, features=features, design_ids=np.arange(4), provenance={'fit_id': 'toy'})
    restored, rows, ids, meta = load_prediction_state(path)
    points = np.array([[.2], [.7]])
    for field, values in model.predict(points).items():
        np.testing.assert_allclose(restored.predict(points)[field], values, rtol=5e-12, atol=5e-13)
    joint = np.zeros((6, 3, 6, 3))
    joint[:4, :, :4, :] = noise
    joint[4:, :, :4, :] = .0001
    joint[:4, :, 4:, :] = .0001
    joint[4:, :, 4:, :] = np.eye(6).reshape(2, 3, 2, 3)*.02
    for field, values in model.predict(points, joint).items():
        np.testing.assert_allclose(restored.predict(points, joint)[field], values, rtol=5e-12, atol=5e-13)
    assert meta['provenance']['fit_id'] == 'toy'
    assert rows.shape == (3, 3) and ids.tolist() == list(range(4))
    with pytest.raises(FileExistsError):
        save_prediction_state(path, model, features=features, design_ids=np.arange(4), provenance={})
    before = path.read_bytes()
    with pytest.raises(ValueError, match='explicit .npz'):
        save_prediction_state(path.with_suffix(''), model, features=features,
                              design_ids=np.arange(4), provenance={})
    assert path.read_bytes() == before


def test_invalid_checkpoint_fields_rejected(tmp_path):
    path = tmp_path/'bad.npz'
    np.savez(path, metadata=np.array('{}'))
    with pytest.raises(ValueError, match='fields'):
        load_prediction_state(path)


@pytest.mark.parametrize('record', [dict(kind='exec', code='bad'),
    dict(kind='constant', value=-1), dict(kind='matern', length_scale=[0], nu=1.5),
    dict(kind='matern', length_scale=[1], nu=0)])
def test_unknown_or_invalid_kernels_rejected(record):
    with pytest.raises(ValueError):
        restore_kernel(record)

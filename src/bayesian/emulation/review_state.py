"""Portable, non-pickle prediction state for an explicitly frozen MeanAware GP.

This is a prediction checkpoint, not a replacement for training data/noise or a
refitting recipe. It cannot certify the provenance or calibration of its input.
Only explicitly supported kernel families are deserialized; no code is loaded.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, Product, RBF

from bayesian.emulation.mean_aware import FastMeanAware

FIELDS = ('projection', 'transfer', 'center', 'residual', 'omitted_projection',
          'omitted_transfer', 'inverse', 'S', 'L', 'H', 'y', 'SinvH', 'mean_L',
          'beta', 'alpha', 'mean_covariance', 'omitted_covariance')
SCHEMA = 'bayesian-mean-aware-prediction-v1'


def kernel_record(kernel):
    if type(kernel) is Product:
        return dict(kind='product', left=kernel_record(kernel.k1), right=kernel_record(kernel.k2))
    if type(kernel) is ConstantKernel:
        return dict(kind='constant', value=float(kernel.constant_value))
    if type(kernel) in (Matern, RBF):
        value = dict(kind='matern' if type(kernel) is Matern else 'rbf',
                     length_scale=np.asarray(kernel.length_scale).tolist())
        if type(kernel) is Matern:
            value['nu'] = float(kernel.nu)
        return value
    raise ValueError('Unsupported prediction-state kernel: ' + type(kernel).__name__)


def restore_kernel(record, depth=0):
    if depth > 8 or not isinstance(record, dict):
        raise ValueError('Invalid prediction-state kernel')
    kind = record.get('kind')
    if kind == 'product' and set(record) == {'kind', 'left', 'right'}:
        return restore_kernel(record['left'], depth+1) * restore_kernel(record['right'], depth+1)
    if kind == 'constant' and set(record) == {'kind', 'value'}:
        value = float(record['value'])
        if not np.isfinite(value) or value <= 0:
            raise ValueError('Invalid constant kernel')
        return ConstantKernel(value, constant_value_bounds='fixed')
    expected = {'kind', 'length_scale'} | ({'nu'} if kind == 'matern' else set())
    if kind in ('matern', 'rbf') and set(record) == expected:
        length = np.asarray(record['length_scale'], dtype=float)
        if length.ndim > 1 or not length.size or not np.isfinite(length).all() or np.any(length <= 0):
            raise ValueError('Invalid kernel length scale')
        if kind == 'rbf':
            return RBF(length, length_scale_bounds='fixed')
        nu = float(record['nu'])
        if nu not in (.5, 1.5, 2.5):
            raise ValueError('Unsupported Matern smoothness')
        return Matern(length, length_scale_bounds='fixed', nu=nu)
    raise ValueError('Unknown or malformed kernel record')


def save_prediction_state(path, model, *, features, design_ids, provenance):
    """Export an already fitted trusted model; never optimize or load a pickle."""
    path = Path(path)
    if path.suffix != '.npz':
        raise ValueError('Prediction checkpoints require an explicit .npz filename')
    if path.exists():
        raise FileExistsError('Do not overwrite a frozen prediction checkpoint')
    meta = dict(schema=SCHEMA, n=model.n, f=model.f, p=model.p, o=model.o,
                kernels=[kernel_record(k) for k in model.gp.kernels_],
                jitter=float(model.gp.jitter), provenance=provenance,
                scope='Frozen latent/noisy-target prediction only; not a training-data archive')
    arrays = {name: np.asarray(getattr(model, name)) for name in FIELDS}
    arrays.update(X_train=model.gp.X_train_, Y_train=model.gp.Y_train_,
                  features=np.asarray(features, dtype=str), design_ids=np.asarray(design_ids),
                  metadata=np.array(json.dumps(meta, allow_nan=False)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)


def load_prediction_state(path, *, schema=SCHEMA):
    """Load numerical state, rejecting missing/extra fields and invalid shapes."""
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(FIELDS) | {'X_train', 'Y_train', 'features', 'design_ids', 'metadata'}:
            raise ValueError('Prediction checkpoint fields differ from schema')
        data = {key: archive[key].copy() for key in archive.files}
    meta = json.loads(str(data.pop('metadata')))
    if not isinstance(schema, str) or not schema or meta.get('schema') != schema:
        raise ValueError('Unknown prediction checkpoint schema')
    n, f, p, o = [meta.get(key) for key in ('n', 'f', 'p', 'o')]
    if any(type(x) is not int or x < 0 for x in (n, f, p, o)) or not n or not p or p+o != f:
        raise ValueError('Invalid prediction checkpoint dimensions')
    s = p*n+o
    shapes = dict(projection=(p, f), transfer=(p, f), center=(f,), residual=(f, f),
                  omitted_projection=(o, f), omitted_transfer=(o, f), inverse=(f, f),
                  S=(s, s), L=(s, s), H=(s, f), y=(s,), SinvH=(s, f),
                  mean_L=(f, f), beta=(f,), alpha=(s,), mean_covariance=(f, f),
                  omitted_covariance=(o, o), Y_train=(n, p), features=(f, 3), design_ids=(n,))
    for name, shape in shapes.items():
        if data[name].shape != shape:
            raise ValueError('Invalid checkpoint shape: ' + name)
    for name in (*FIELDS, 'X_train', 'Y_train'):
        if data[name].dtype.kind not in 'fiu' or not np.isfinite(data[name]).all():
            raise ValueError('Invalid numerical checkpoint field: ' + name)
    x = data['X_train']
    if x.ndim != 2 or len(x) != n or not x.shape[1]:
        raise ValueError('Invalid training coordinates')
    if data['design_ids'].dtype.kind not in 'iu' or len(set(data['design_ids'].tolist())) != n:
        raise ValueError('Invalid design identities')
    if len(set(map(tuple, data['features']))) != f:
        raise ValueError('Duplicate feature identities')
    edges = data['features'][:, 1:].astype(float)
    if not np.isfinite(edges).all() or np.any(edges[:, 1] <= edges[:, 0]):
        raise ValueError('Invalid feature bin edges')
    for name in ('L', 'mean_L'):
        if np.any(np.diag(data[name]) <= 0) or np.any(np.triu(data[name], 1)):
            raise ValueError('Invalid covariance factor: ' + name)
    np.testing.assert_allclose(data['L'] @ data['L'].T, data['S'], rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(np.vstack([data['projection'], data['omitted_projection']])
                               @ data['inverse'].T, np.eye(f), rtol=1e-9, atol=1e-10)
    if not isinstance(meta.get('kernels'), list) or len(meta['kernels']) != p:
        raise ValueError('Invalid kernel count')
    kernels = [restore_kernel(row) for row in meta['kernels']]
    jitter = float(meta['jitter'])
    if not np.isfinite(jitter) or jitter < 0:
        raise ValueError('Invalid conditioning jitter')
    model = FastMeanAware.__new__(FastMeanAware)
    for key in FIELDS:
        setattr(model, key, data[key])
    model.n, model.f, model.p, model.o = n, f, p, o
    model.gp = SimpleNamespace(X_train_=x, Y_train_=data['Y_train'], kernels_=kernels, jitter=jitter)
    return model, data['features'], data['design_ids'], meta

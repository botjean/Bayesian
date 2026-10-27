"""Small fresh-data training tests, never author-workspace state replay."""
from copy import deepcopy
import json

import numpy as np
import pytest
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from threadpoolctl import threadpool_limits
import yaml

from bayesian import data_IO
from bayesian.emulation import base, joint_groups as jg, training as pt
from bayesian.emulation.mean_aware import MeanAware, RestrictedMeanGP
from bayesian.emulation.review_state import load_prediction_state


@pytest.fixture(autouse=True)
def one_blas_thread():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture
def inputs(tmp_path):
    hadron = '200__PbPb__hadron__pt__charged__0-10'
    jet = '200__PbPb__jet__pt__inclusive__0-10'
    names = [hadron, jet]
    x = np.linspace(.04, .96, 6)[:, None]
    xv = np.array([[.43]])
    raw = dict(Design=x, Design_validation=xv, Design_indices=np.arange(6),
               Design_indices_validation=np.array([71]), Data={}, Prediction={}, Prediction_validation={})
    for j, name in enumerate(names):
        edges = np.arange(9, dtype=float)
        raw['Data'][name] = dict(xmin=edges, xmax=edges + 1, y=edges + 1,
                                 y_err=np.full(9, .2), metadata={'source': 'toy'})
        for section, points in [('Prediction', x), ('Prediction_validation', xv)]:
            y = np.array([2 + j + np.sin((i + 1) * points[:, 0]) + .13 * i * points[:, 0]**2
                          for i in range(9)])
            raw[section][name] = dict(y=y, y_err=np.full_like(y, .01))
    raw['correlation_manager'] = dict(_observable_pt={name: np.arange(9.) / 8 for name in names},
                                      observable_systematics={name: ['test'] for name in names})
    path = tmp_path / 'full.h5'
    data_IO.write_dict_to_h5(raw, tmp_path, path.name, verbose=False)
    groups = {}
    # Group order deliberately differs from native feature order.
    for group, name, seed in [('jet', jet, 19), ('hadron', hadron, 35)]:
        groups[group] = dict(emulator_package='sk_learn', force_retrain=False, n_pc=1,
            random_state=seed, pca_whiten=True, observable_list=[{'observable': name}],
            kernels=dict(active=['matern', 'signal_amplitude'],
                matern=dict(nu=1.5, length_scale_bounds_factor=[.03, 4]),
                signal_amplitude=dict(constant_value_factor=1., constant_value_bounds_factor=[.01, 100.])),
            GPR=dict(n_restarts=2, alpha=1e-8),
            simulation_noise=dict(covariance_file='/unused/original.npz',
                                  joint_groups=True, residual_white_noise=False))
    cfg = dict(observable_table_dir='/unused/tables', observable_config_dir='/unused/config',
               observables_filename='old.h5', output_dir='/unused/frozen',
               analyses={'analysis_test': dict(parameterizations=['unit'],
                   parameterization={'unit': dict(names=['x'], min=[0.], max=[1.])},
                   parameters=dict(emulators=groups))})
    features = [(name, float(i), float(i + 1)) for name in names for i in range(9)]
    wanted = [row for row in features if (row[0] == hadron and int(row[1]) in [0, 2, 3, 4, 5, 7, 8])
              or (row[0] == jet and int(row[1]) in [1, 6])]
    rng = np.random.RandomState(72)
    factor = rng.normal(size=(7 * 18, 4)) * .008
    covariance = (factor @ factor.T + np.eye(7 * 18) * .002).reshape(7, 18, 7, 18)
    # Permute both identity axes to ensure that the native identity loader is used.
    di, fi = [6, 2, 0, 5, 1, 4, 3], np.arange(18)[::-1]
    covpath = tmp_path / 'noise.npz'
    ids = np.r_[np.arange(6), 71]
    np.savez(covpath, design_ids=ids[di], observable_names=np.array([r[0] for r in features])[fi],
             xmin=np.array([r[1] for r in features])[fi], xmax=np.array([r[2] for r in features])[fi],
             covariance=covariance[np.ix_(di, fi, di, fi)])
    return dict(path=path, config=cfg, covariance_path=covpath, selected=wanted,
                raw=raw, full_features=features, full_covariance=covariance)


def call(function, inputs, output, **kwargs):
    return function(inputs['path'], inputs['config'], inputs['covariance_path'],
                    inputs['selected'], output, **kwargs)


def test_prepare_native_parity_and_real_selected_training(inputs, tmp_path, monkeypatch):
    original_cfg = deepcopy(inputs['config'])
    def forbidden(*args, **kwargs):
        raise AssertionError('prepare must not fit or read a model')
    monkeypatch.setattr(RestrictedMeanGP, 'fit', forbidden)
    monkeypatch.setattr(base.IO, 'read_emulator', forbidden)
    bank = call(pt.prepare, inputs, tmp_path / 'prepared')
    assert inputs['config'] == original_cfg
    assert bank['features'] == inputs['selected']
    assert bank['grouped_features'] != bank['features']
    assert len(bank['features']) == 9
    assert bank['targets'].shape == (6, 2)
    for key in ('Design', 'Design_validation', 'Design_indices', 'Design_indices_validation'):
        np.testing.assert_array_equal(bank['raw'][key], inputs['raw'][key])
    for section in ('Data', 'Prediction', 'Prediction_validation'):
        for name in bank['raw'][section]:
            ix = [int(lo) for obs, lo, _ in bank['features'] if obs == name]
            for key in ('y', 'y_err'):
                np.testing.assert_array_equal(bank['raw'][section][name][key], inputs['raw'][section][name][key][ix])
    fi = [inputs['full_features'].index(row) for row in bank['grouped_features']]
    noise = inputs['full_covariance'][:6, fi, :6, :][:, :, :, fi]
    np.testing.assert_array_equal(bank['grouped_noise'], noise)
    transform = np.kron(np.eye(6), bank['projection'])
    reorder = np.arange(12).reshape(6, 2).T.ravel()
    expected = transform @ noise.reshape(54, 54) @ transform.T
    np.testing.assert_allclose(bank['noise_pc'], expected[np.ix_(reorder, reorder)], atol=1e-12)
    blocks = []
    for group, es in bank['ec'].emulation_settings.items():
        result = bank['models'][group]
        pca = result['PCA']
        scores = pca['Y_pca_truncated'][:, 0]
        v = np.var(scores)
        expected_kernel = ConstantKernel(v, (v * .01, v * 100)) * Matern(1., (.03, 4.), nu=1.5)
        actual = result['_initial_kernels'][0]
        np.testing.assert_allclose(actual.theta, expected_kernel.theta)
        np.testing.assert_allclose(actual.bounds, expected_kernel.bounds)
        bounds = np.vstack([k.bounds for k in result['_initial_kernels']])
        rng = np.random.RandomState(es.settings['random_state'])
        blocks.append([rng.uniform(bounds[:, 0], bounds[:, 1]) for _ in range(2)])
    for i in range(2):
        np.testing.assert_array_equal(bank['restart_thetas'][i], np.concatenate([b[i] for b in blocks]))
    maps = bank['maps']
    # Seven hadron bins, but only six SVD rows: the absent direction is included.
    assert maps['omitted_projection'].shape == (7, 9)
    np.testing.assert_allclose(np.vstack([maps['projection'], maps['omitted_projection']]) @
                              np.vstack([maps['transfer'], maps['omitted_transfer']]).T, np.eye(9), atol=1e-12)
    assert not list((tmp_path / 'prepared').rglob('*.pkl'))


@pytest.mark.parametrize('whiten', [False, True])
def test_train_fresh_roundtrip_and_paired_holdout(inputs, tmp_path, monkeypatch, whiten):
    for group in inputs['config']['analyses']['analysis_test']['parameters']['emulators'].values():
        group['pca_whiten'] = whiten
    def forbidden(*args, **kwargs):
        raise AssertionError('Training cannot read prior emulator pickles')
    monkeypatch.setattr(base.IO, 'read_emulator', forbidden)
    before = [pt._sha(inputs[key]) for key in ('path', 'covariance_path')]
    out = tmp_path / 'trained'
    bank = call(pt.train, inputs, out, optimize=False)
    assert bank['report']['optimize'] is False
    assert bank['report']['status'] == 'TRAINED_UNOPTIMIZED'
    assert set(['model', 'features', 'raw', 'models', 'ec', 'settings', 'gp', 'report']) <= bank.keys()
    jg.validate_results(bank['ec'], bank['models'], check_file=True)
    assert [pt._sha(inputs[key]) for key in ('path', 'covariance_path')] == before
    points = np.array([[.23], [.67]])
    fast, slow = bank['model'].predict(points), MeanAware.predict(bank['model'], points)
    restored, features, ids, meta = load_prediction_state(out / 'prediction_state.npz')
    for key in fast:
        np.testing.assert_allclose(fast[key], slow[key], atol=2e-11, rtol=1e-9)
        np.testing.assert_allclose(fast[key], restored.predict(points)[key], atol=1e-13, rtol=1e-11)
    assert len(features) == 9
    np.testing.assert_array_equal(ids, np.arange(6))
    assert meta['provenance']['holdout_ids'] == [71]
    fi = [inputs['full_features'].index(row) for row in bank['features']]
    noise = inputs['full_covariance'][:, fi, :, :][:, :, :, fi]
    heldout = bank['model'].predict(inputs['raw']['Design_validation'], noise)
    with np.load(out / 'holdout_qa.npz', allow_pickle=False) as qa:
        np.testing.assert_array_equal(qa['observed_mean'], heldout['central_value'])
        np.testing.assert_allclose(qa['observed_covariance'], heldout['cov'], atol=1e-13, rtol=1e-11)
        assert abs(qa['observed_covariance'] - qa['latent_covariance']).max() > 1e-7
    assert len(list(bank['settings'].output_dir.glob('*.pkl'))) == 2
    report = json.loads((out / 'training_report.json').read_text())
    assert np.isfinite(report['objective'])
    assert report['holdout_qa']['paired_mc_available']


def test_default_optimizes_with_explicit_native_restarts(inputs, tmp_path, monkeypatch, capsys):
    actual_fit = RestrictedMeanGP.fit
    seen = []
    def fit(self, x, y, noise, **kwargs):
        seen.append(kwargs)
        return actual_fit(self, x, y, noise, **kwargs)
    monkeypatch.setattr(RestrictedMeanGP, 'fit', fit)
    bank = call(pt.train, inputs, tmp_path / 'optimized', overrides={'n_restarts': 0})
    assert seen[0]['optimize'] is True
    assert seen[0]['restart_thetas'] == []
    assert bank['report']['best_success']
    assert len(bank['report']['optimization']) == 1
    assert bank['report']['status'] == 'TRAINED'
    assert not np.array_equal(bank['report']['initial_theta'], bank['report']['fitted_theta'])
    output = capsys.readouterr().out
    assert 'REML_FIT_BEGIN' in output and 'REML_PROGRESS start=1/1' in output and 'REML_FIT_END' in output


def test_failed_best_optimizer_saves_diagnostics_not_trained_state(inputs, tmp_path, monkeypatch):
    actual_fit = RestrictedMeanGP.fit
    def fit(self, x, y, noise, **kwargs):
        result = actual_fit(self, x, y, noise, optimize=False)
        self.optimization_ = [dict(success=False, message='test optimizer failure', objective=1.)]
        return result
    monkeypatch.setattr(RestrictedMeanGP, 'fit', fit)
    out = tmp_path / 'failed'
    with pytest.raises(RuntimeError, match='did not converge'):
        call(pt.train, inputs, out)
    reports = list(out.rglob('joint_fit.json'))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert report['status'] == 'OPTIMIZATION_FAILED'
    assert not list(out.rglob('*.pkl'))
    assert not (out / 'prediction_state.npz').exists()


def test_noncontiguous_training_ids_are_preserved(inputs, tmp_path):
    raw = deepcopy(inputs['raw'])
    ids = 2 + 3 * np.arange(12)
    x = np.linspace(.01, .99, 12)[:, None]
    raw['Design'], raw['Design_indices'] = x, ids
    for name, values in raw['Prediction'].items():
        values['y'] = np.array([2 + np.sin((i + 1) * x[:, 0]) + .1 * i * x[:, 0]**2 for i in range(9)])
        values['y_err'] = np.full((9, 12), .01)
    source = tmp_path / 'noncontiguous_designs.h5'
    data_IO.write_dict_to_h5(raw, tmp_path, source.name, verbose=False)
    inputs['path'] = source
    inputs['selected'] = [inputs['full_features'][0], inputs['full_features'][9]]
    np.savez(tmp_path / 'noncontiguous_noise.npz', design_ids=np.r_[ids, 71],
             observable_names=[r[0] for r in inputs['selected']], xmin=[0., 0.], xmax=[1., 1.],
             covariance=np.eye(26).reshape(13, 2, 13, 2) * .001)
    inputs['covariance_path'] = tmp_path / 'noncontiguous_noise.npz'
    cfg = inputs['config']['analyses']['analysis_test']
    cfg['validation_indices'] = [71, 72]
    bank = call(pt.prepare, inputs, tmp_path / 'preserve_ids')
    np.testing.assert_array_equal(bank['x'], x)
    assert bank['report']['training_ids'] == ids.tolist()
    assert bank['report']['holdout_ids'] == [71]
    assert bank['targets'].shape == (12, 2)


def test_filter_group_entries_drop_unselected_group_and_preserve_settings(inputs, tmp_path):
    original = deepcopy(inputs['config'])
    inputs['selected'] = [row for row in inputs['selected'] if '__hadron__' in row[0]]
    bank = call(pt.prepare, inputs, tmp_path / 'hadrons')
    assert list(bank['ec'].emulation_settings) == ['hadron']
    cfg = bank['config']['analyses']['analysis_test']['parameters']['emulators']['hadron']
    before = original['analyses']['analysis_test']['parameters']['emulators']['hadron']
    for key in ('kernels', 'GPR', 'random_state', 'n_pc', 'pca_whiten'):
        assert cfg[key] == before[key]
    assert inputs['config'] == original


def test_missing_holdout_noise_keeps_latent_qa(inputs, tmp_path):
    path = tmp_path / 'training_only_noise.npz'
    with np.load(inputs['covariance_path'], allow_pickle=False) as archive:
        ix = np.flatnonzero(archive['design_ids'] != 71)
        np.savez(path, **{key: archive[key] for key in ('observable_names', 'xmin', 'xmax')},
                 design_ids=archive['design_ids'][ix], covariance=archive['covariance'][ix, :, :, :][:, :, ix, :])
    inputs['covariance_path'] = path
    bank = call(pt.train, inputs, tmp_path / 'latent_qa', optimize=False)
    assert bank['report']['holdout_qa'] == dict(available=True, paired_mc_available=False)
    with np.load(tmp_path / 'latent_qa' / 'holdout_qa.npz', allow_pickle=False) as qa:
        assert 'latent_covariance' in qa and 'observed_covariance' not in qa


def test_yaml_overrides_local_config_and_no_overwrite(inputs, tmp_path):
    path = tmp_path / 'template.yaml'
    path.write_text(yaml.safe_dump(inputs['config']))
    inputs['config'] = path
    output = tmp_path / 'selected'
    bank = call(pt.prepare, inputs, output,
                overrides={'n_restarts': 0, 'random_state': {'hadron': 7}, 'n_pc': {'hadron': 2}})
    assert bank['ec'].emulation_settings['hadron'].n_pc == 2
    assert bank['report']['seeds'] == {'hadron': 7, 'jet': 19}
    cfg = yaml.safe_load((output / 'config.yaml').read_text())
    assert cfg['observable_table_dir'] == str(output / 'inputs')
    assert cfg['output_dir'] == str(output / 'results')
    assert cfg['observables_filename'] == 'observables.h5'
    before = pt._sha(output / 'config.yaml')
    with pytest.raises(FileExistsError, match='NEW'):
        call(pt.train, inputs, output, optimize=False)
    assert pt._sha(output / 'config.yaml') == before


@pytest.mark.parametrize('case', ['duplicate', 'missing', 'pc', 'white', 'coverage', 'override'])
def test_invalid_training_inputs_fail_closed(inputs, tmp_path, case):
    groups = inputs['config']['analyses']['analysis_test']['parameters']['emulators']
    kwargs = {}
    if case == 'duplicate':
        inputs['selected'].append(inputs['selected'][0])
    elif case == 'missing':
        inputs['selected'][0] = (inputs['selected'][0][0], -1., 0.)
    elif case == 'pc':
        groups['hadron']['n_pc'] = 20
    elif case == 'white':
        groups['hadron']['kernels']['active'].append('noise')
    elif case == 'coverage':
        groups['jet']['observable_list'].extend(groups['hadron']['observable_list'])
    else:
        kwargs['overrides'] = {'noise_inflation': 2}
    with pytest.raises(ValueError):
        call(pt.prepare, inputs, tmp_path / 'invalid', **kwargs)
    assert not (tmp_path / 'invalid').exists()

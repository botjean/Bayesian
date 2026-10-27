"""Cross-group known noise survives fitting, cache IO and native prediction."""
from types import SimpleNamespace
import copy

import numpy as np
import pytest
from scipy.linalg import block_diag

from bayesian.emulation import base, interface, joint_groups as jg, sk_learn, simulation_noise as sn
from tests.test_emulation_signal_amplitude import configuration


@pytest.fixture(params=[False, True], ids=['unwhitened', 'whitened'])
def fitted(tmp_path, monkeypatch, request):
    x = np.linspace(0, 1, 7)[:, None]
    ya = np.column_stack((np.sin(3*x[:, 0]), np.cos(2*x[:, 0])))
    yb = np.column_stack((2*np.sin(2*x[:, 0])+1, 3*np.cos(x[:, 0])))
    rng = np.random.default_rng(35)
    factor = rng.normal(size=(28, 5))*.04
    noise = factor @ factor.T + np.eye(28)*.01
    path = tmp_path/'cov.npz'
    np.savez(path, design_ids=np.arange(7), observable_names=['a','a','b','b'],
             xmin=[0.,1.,0.,1.], xmax=[1.,2.,1.,2.], covariance=noise.reshape(7,4,7,4))
    settings = {}
    for name in ['a', 'b']:
        c = configuration()
        c['pca_whiten'] = request.param
        c['kernels']['active'].remove('noise')
        c['additional_name'] = name
        c['observable_list'] = [name]
        c['random_state'] = 42
        c['simulation_noise'] = dict(covariance_file=str(path), residual_white_noise=False, joint_groups=True)
        settings[name] = sk_learn.SKLearnEmulatorSettings.from_config(c)
    filters = {id(s.base_settings.observable_filter): name for name, s in settings.items()}
    raw = {'Design_indices': np.arange(7), 'Data': {name: dict(xmin=np.array([0.,1.]), xmax=np.array([1.,2.])) for name in settings}}
    monkeypatch.setattr(sk_learn.data_IO, 'predictions_matrix_from_h5',
                        lambda **k: {'a': ya, 'b': yb}[filters[id(k['observable_filter'])]])
    monkeypatch.setattr(sk_learn.data_IO, 'design_array_from_h5', lambda *a, **k: x)
    monkeypatch.setattr(sk_learn.data_IO, 'read_dict_from_h5', lambda *a, **k: raw)
    monkeypatch.setattr(sk_learn.data_IO, 'sorted_observable_list_from_dict',
                        lambda *a, **k: [filters[id(k['observable_filter'])]])
    analysis = SimpleNamespace(output_dir=tmp_path, io=SimpleNamespace(observables_filename='test'),
                               parameterization='toy', raw_analysis_config={'parameterization': {'toy': {'min':[0.], 'max':[1.]}}})
    # Deliberately reverse the output group order to catch accidental block ordering.
    mapping = SimpleNamespace(shape=(7,4), emulation_group_to_observable_matrix={
        'a': ('a', slice(2,4), slice(0,2)), 'b': ('b', slice(0,2), slice(0,2))})
    config = SimpleNamespace(emulation_settings=settings, sort_observables_in_matrix=mapping)
    interface.fit_emulators(config, analysis)
    models = {name: base.IO.read_emulator(s, analysis) for name, s in settings.items()}
    return config, analysis, models, noise, x


@pytest.mark.filterwarnings('ignore::sklearn.exceptions.ConvergenceWarning')
def test_native_joint_fit_projection_and_predictions(fitted):
    config, analysis, models, noise, x = fitted
    joint = models['a']['joint_pc_gp'].joint
    projections = [r['PCA']['pca'].components_ / r['PCA']['scaler'].scale_ for r in models.values()]
    projections = [p / np.sqrt(r['PCA']['pca'].explained_variance_[:, None])
                   if r['PCA']['pca'].whiten else p
                   for p, r in zip(projections, models.values())]
    design_major = np.kron(np.eye(7), block_diag(*projections))
    reorder = np.arange(28).reshape(7,4).T.ravel()
    expected_noise = (design_major @ noise @ design_major.T)[np.ix_(reorder,reorder)]
    np.testing.assert_allclose(joint.noise_, expected_noise, atol=1e-12)
    points = np.array([[.2], [.7]])
    merged = interface.predict(points, config, analysis_settings=analysis, emulator_results=models)
    pc_mean, pc_cov = joint.predict(points)
    transfer = block_diag(*[r['PCA']['transfer']*r['PCA']['scaler'].scale_ for r in models.values()])
    out_order = [2,3,0,1]
    direct_mean = pc_mean @ transfer + np.concatenate([r['PCA']['scaler'].mean_ for r in models.values()])
    direct_cov = np.array([transfer.T @ c @ transfer for c in pc_cov])
    np.testing.assert_allclose(merged['central_value'], direct_mean[:,out_order], atol=1e-12)
    np.testing.assert_allclose(merged['cov'], direct_cov[:,out_order][:,:,out_order], atol=1e-12)
    assert abs(merged['cov'][:,:2,2:]).max() > 1e-6
    assert np.linalg.eigvalsh(merged['cov']).min() > -1e-12
    group = interface.predict(points, config, analysis_settings=analysis, emulator_results=models, merge_predictions_over_groups=False)
    native_a = sk_learn.predict(points, models['a'], config.emulation_settings['a'])
    for key in ['central_value', 'cov']:
        np.testing.assert_allclose(group['a'][key], native_a[key], atol=1e-12)
    for i in range(2):
        single = interface.predict(points[i:i+1], config, analysis_settings=analysis, emulator_results=models)
        np.testing.assert_allclose(single['cov'][0], merged['cov'][i], atol=1e-12)
    interface.fit_emulators(config, analysis)  # Cache validation, not another fit.
    restored = interface.predict(points, config, analysis_settings=analysis)
    np.testing.assert_array_equal(restored['cov'], merged['cov'])
    partial_qa = interface.predict(points, config, analysis_settings=analysis,
                                   emulator_results={'a': models['a']}, merge_predictions_over_groups=False)
    np.testing.assert_array_equal(partial_qa['a']['cov'], group['a']['cov'])


@pytest.mark.filterwarnings('ignore::sklearn.exceptions.ConvergenceWarning')
def test_mixed_caches_and_partial_opt_in_rejected(fitted):
    config, analysis, models, _, _ = fitted
    broken = copy.deepcopy(models)
    broken['a']['joint_group_fit_id'] = 'wrong-fit'
    with pytest.raises(ValueError, match='Mixed'):
        jg.validate_results(config, broken)
    with pytest.raises(ValueError, match='all jointly fitted'):
        jg.validate_results(config, {'a': models['a']})
    config.emulation_settings['a'].settings['simulation_noise']['joint_groups'] = False
    with pytest.raises(ValueError, match='all selected groups'):
        jg.options(config)


def test_independent_settings_do_not_enable_joint_mode():
    settings = sk_learn.SKLearnEmulatorSettings.from_config(configuration())
    assert not jg.enabled(SimpleNamespace(emulation_settings={'a': settings}))


def test_invalid_restart_vectors_rejected():
    from sklearn.gaussian_process.kernels import RBF
    x = np.linspace(0,1,3)[:,None]
    with pytest.raises(ValueError, match='restarts'):
        sn.JointPCGP([RBF()], n_restarts=1).fit(x, x, np.eye(3), restart_thetas=[])

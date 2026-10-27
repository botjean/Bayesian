"""Opt-in joint noise conditioning with separate, unchanged group PCA bases."""
from __future__ import annotations

import uuid
import numpy as np
from scipy.linalg import block_diag

from bayesian import data_IO
from bayesian.emulation import base, simulation_noise as sn


def enabled(config):
    return any(getattr(s, 'settings', {}).get('simulation_noise', {}).get('joint_groups', False)
               for s in config.emulation_settings.values())


def options(config):
    settings = list(config.emulation_settings.values())
    configs = [sn.configuration(s) for s in settings]
    if not configs or any(s.emulator_name != 'sk_learn' or c is None or not c.get('joint_groups')
                          for s, c in zip(settings, configs)):
        raise ValueError('Joint groups require all selected groups to opt in using sk_learn')
    if any(c != configs[0] for c in configs):
        raise ValueError('Joint groups must use the same simulation covariance and residual-noise mode')
    if len({(s.alpha, s.n_restarts) for s in settings}) != 1:
        raise ValueError('Joint groups require matching alpha and restart counts')
    return configs[0]


class GroupView:
    """Native single-group QA view of the joint conditional distribution."""
    def __init__(self, joint, indices):
        self.joint = joint
        self.indices = np.asarray(indices, dtype=int)

    def predict(self, points):
        mean, cov = self.joint.predict(points)
        return mean[:, self.indices], cov[:, self.indices][:, :, self.indices]


def validate_results(config, results, check_file=False):
    options(config)
    groups = tuple(config.emulation_settings)
    if set(results) != set(groups):
        raise ValueError('Joint prediction requires all jointly fitted group caches')
    ids = set()
    indices = []
    for name, settings in config.emulation_settings.items():
        result = results[name]
        sn.validate_cache(result, settings, check_file=check_file)
        if tuple(result.get('joint_group_names', ())) != groups or not isinstance(result.get('joint_pc_gp'), GroupView):
            raise ValueError('Joint group cache membership changed; retrain')
        ids.add(result.get('joint_group_fit_id'))
        indices.extend(result['joint_pc_gp'].indices.tolist())
    if len(ids) != 1 or None in ids or indices != list(range(sum(s.n_pc for s in config.emulation_settings.values()))):
        raise ValueError('Mixed or misaligned joint group caches; retrain')


def fit(config, analysis_settings):
    from bayesian.emulation import sk_learn
    choice = options(config)
    filenames = [base.IO.output_filename(s, analysis_settings) for s in config.emulation_settings.values()]
    if all(path.exists() for path in filenames) and not any(s.force_retrain for s in config.emulation_settings.values()):
        results = {name: base.IO.read_emulator(s, analysis_settings) for name, s in config.emulation_settings.items()}
        validate_results(config, results, check_file=True)
        return
    if any(path.exists() for path in filenames) and not all(s.force_retrain for s in config.emulation_settings.values()):
        raise ValueError('Partial joint cache: explicitly force retraining all groups')
    prepared = {name: sk_learn.fit_emulator(s, analysis_settings, prepare_only=True)
                for name, s in config.emulation_settings.items()}
    raw = data_IO.read_dict_from_h5(analysis_settings.output_dir, analysis_settings.io.observables_filename, verbose=False)
    features, projections, kernels, targets, restart_blocks = [], [], [], [], []
    x = next(iter(prepared.values()))['_training_design']
    first_settings = next(iter(config.emulation_settings.values()))
    for name, settings in config.emulation_settings.items():
        result = prepared[name]
        np.testing.assert_array_equal(result['_training_design'], x)
        names = data_IO.sorted_observable_list_from_dict(raw, observable_filter=settings.base_settings.observable_filter)
        group_features = [(key, float(lo), float(hi)) for key in names
                          for lo, hi in zip(raw['Data'][key]['xmin'], raw['Data'][key]['xmax'], strict=True)]
        features.extend(group_features)
        pca, scaler = result['PCA']['pca'], result['PCA']['scaler']
        assert len(group_features) == len(scaler.scale_)
        projection = pca.components_[:settings.n_pc] / scaler.scale_
        if pca.whiten:
            projection = projection / np.sqrt(pca.explained_variance_[:settings.n_pc, None])
        projections.append(projection)
        group_kernels = result['_initial_kernels']
        kernels.extend(group_kernels)
        targets.append(result['PCA']['Y_pca_truncated'])
        bounds = np.vstack([k.bounds for k in group_kernels])
        rng = np.random.RandomState(settings.settings.get('random_state'))
        restart_blocks.append([rng.uniform(bounds[:, 0], bounds[:, 1]) for _ in range(settings.n_restarts)])
    cov = sn.load_aligned_covariance(choice['covariance_file'], raw['Design_indices'], features)
    projection = block_diag(*projections)
    noise = np.einsum('ab,ibjc,dc->aidj', projection, cov, projection, optimize=True).reshape(len(kernels)*len(x), -1)
    restarts = [np.concatenate([block[i] for block in restart_blocks]) for i in range(first_settings.n_restarts)]
    joint = sn.JointPCGP(kernels, jitter=first_settings.alpha, n_restarts=first_settings.n_restarts).fit(
        x, np.concatenate(targets, axis=1), noise, restart_thetas=restarts)
    fit_id, offset = uuid.uuid4().hex, 0
    fingerprint = sn.fingerprint(choice)
    for name, settings in config.emulation_settings.items():
        result = prepared[name]
        indices = np.arange(offset, offset+settings.n_pc)
        result['joint_pc_gp'] = GroupView(joint, indices)
        result['emulators'] = [sn.PCView(joint, i) for i in indices]
        result['simulation_noise'] = fingerprint
        result['joint_group_fit_id'] = fit_id
        result['joint_group_names'] = tuple(config.emulation_settings)
        del result['_training_design'], result['_initial_kernels']
        base.IO.write_emulator(result, settings, analysis_settings)
        offset += settings.n_pc


def linear_maps(config, results, additional=None):
    """Whiten-aware forward/inverse PC maps in native likelihood feature order."""
    from bayesian.emulation import sk_learn
    mapping = config.sort_observables_in_matrix.emulation_group_to_observable_matrix
    f = config.sort_observables_in_matrix.shape[1]
    transfer = np.zeros((sum(s.n_pc for s in config.emulation_settings.values()), f))
    projection = np.zeros_like(transfer)
    center, residual = np.zeros(f), np.zeros((f, f))
    group_indices = {}
    for name, settings in config.emulation_settings.items():
        result = results[name]
        sk_learn._validate_signal_amplitude_cache(result, settings)
        p = result['PCA']
        pc_indices = result['joint_pc_gp'].indices
        local_transfer = p.get('transfer', p['pca'].components_)[:settings.n_pc] * p['scaler'].scale_
        idx = np.full(local_transfer.shape[1], -1, dtype=int)
        for group, out_sl, local_sl in mapping.values():
            if group == name:
                idx[local_sl] = np.arange(out_sl.start, out_sl.stop)
        if np.any(idx < 0):
            raise ValueError('Incomplete joint observable mapping')
        group_indices[name] = idx
        transfer[np.ix_(pc_indices, idx)] = local_transfer
        forward = p['pca'].components_[:settings.n_pc] / p['scaler'].scale_
        if p['pca'].whiten:
            forward = forward / np.sqrt(p['pca'].explained_variance_[:settings.n_pc, None])
        projection[np.ix_(pc_indices, idx)] = forward
        center[idx] = p['scaler'].mean_
        extra = (additional[name] if additional else
                 sk_learn.compute_additional_covariance_contributions(settings, result))
        residual[np.ix_(idx, idx)] = extra * np.outer(p['scaler'].scale_, p['scaler'].scale_)
    return projection, transfer, center, residual, group_indices


def predict(points, config, analysis_settings, results=None, additional=None, merge=True):
    # Native per-group QA supplies one cache; load the remaining members rather
    # than dropping their conditioning data or weakening mixed-fit validation.
    results = dict(results or {})
    for name, settings in config.emulation_settings.items():
        if name not in results:
            results[name] = base.IO.read_emulator(settings, analysis_settings)
    validate_results(config, results)
    joint = next(iter(results.values()))['joint_pc_gp'].joint
    mean_pc, cov_pc = joint.predict(points)
    _, transfer, center, residual, group_indices = linear_maps(config, results, additional)
    mean = mean_pc @ transfer + center
    cov = np.einsum('pa,mpq,qb->mab', transfer, cov_pc, transfer, optimize=True) + residual
    if merge:
        return {'central_value': mean, 'cov': cov}
    return {name: {'central_value': mean[:, idx], 'cov': cov[:, idx][:, :, idx]}
            for name, idx in group_indices.items()}

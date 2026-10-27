"""Selected-table REML training using native emulator configuration APIs.

``prepare`` stages a NEW output directory and does native PCA, but no GP fit.
``train`` calls it, jointly fits the retained scores, and constructs the complete
mean-aware predictor. No prediction checkpoints or model pickles are read.

Feature order in the returned bank is native likelihood order, not caller order;
``grouped_features`` describes the separate PC/noise training layout. Overrides
are limited to ``n_restarts``, ``random_state`` (integers or group mappings), and
``n_pc`` (a group mapping). All effective choices are saved in config.yaml.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
import uuid
import warnings

import numpy as np
from scipy.linalg import block_diag
import yaml

from bayesian import analysis, data_IO
from bayesian.emulation import base, interface, joint_groups as jg, sk_learn
from bayesian.emulation import simulation_noise as sn
from bayesian.emulation.mean_aware import FastMeanAware, RestrictedMeanGP
from bayesian.emulation.omitted_coordinates import omitted_maps
from bayesian.emulation.review_state import save_prediction_state


class _ProgressRestrictedMeanGP(RestrictedMeanGP):
    """Observe the native objective without changing its values or optimizer."""

    def objective(self, theta, gradient=True):
        result = super().objective(theta, gradient=gradient)
        if not hasattr(self, '_progress_started'):
            return result
        self._progress_evaluations += 1
        now = time.monotonic()
        value = float(result[0] if gradient else result)
        initial = next((i for i, row in enumerate(self._progress_initials)
                        if i not in self._progress_seen and np.array_equal(theta, row)), None)
        if initial is not None:
            self._progress_seen.add(initial)
        if initial is not None or now - self._progress_last >= 30:
            self._progress_last = now
            label = f'start={initial + 1}/{len(self._progress_initials)}' if initial is not None else 'evaluating'
            print(f'REML_PROGRESS {label} evaluations={self._progress_evaluations} '
                  f'objective={value:.12g} seconds={now - self._progress_started:.1f}', flush=True)
        return result


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def _features(raw, names):
    return [(name, float(lo), float(hi)) for name in names
            for lo, hi in zip(raw['Data'][name]['xmin'], raw['Data'][name]['xmax'], strict=True)]


def _slice_bins(value, mask):
    if isinstance(value, dict):
        return {key: _slice_bins(item, mask) for key, item in value.items()}
    if isinstance(value, np.ndarray) and value.ndim and len(value) == len(mask):
        return value[mask].copy()
    return deepcopy(value)


def _filter_raw(raw, selected_features):
    wanted = [(str(name), float(lo), float(hi)) for name, lo, hi in selected_features]
    if not wanted or len(set(wanted)) != len(wanted):
        raise ValueError('Selected features must be nonempty and unique')
    if any(not np.isfinite([lo, hi]).all() or hi <= lo for _, lo, hi in wanted):
        raise ValueError('Selected feature edges must be finite and increasing')
    reference = _features(raw, raw['Data'])
    if len(set(reference)) != len(reference) or not set(wanted) <= set(reference):
        raise ValueError('Selected features are absent or source feature identities are duplicated')
    selected = set(wanted)
    masks = {name: np.array([(name, float(lo), float(hi)) in selected
                            for lo, hi in zip(data['xmin'], data['xmax'], strict=True)])
             for name, data in raw['Data'].items()}
    masks = {name: mask for name, mask in masks.items() if mask.any()}
    result = deepcopy(raw)
    for section in ('Data', 'Prediction', 'Prediction_validation'):
        result[section] = {name: _slice_bins(raw[section][name], mask) for name, mask in masks.items()}
    manager = result.get('correlation_manager', {})
    for key in ('_observable_pt', 'observable_systematics'):
        if key in manager:
            manager[key] = {name: (_slice_bins(item, masks[name]) if key == '_observable_pt' else item)
                            for name, item in manager[key].items() if name in masks}
    for suffix, section in (('', 'Prediction'), ('_validation', 'Prediction_validation')):
        design, ids = raw['Design' + suffix], raw['Design_indices' + suffix]
        if (np.ndim(ids) != 1 or np.asarray(ids).dtype.kind not in 'iu'
                or len(set(ids)) != len(ids) or len(design) != len(ids)):
            raise ValueError('Invalid design identities or row alignment')
        for name, mask in masks.items():
            if np.shape(raw[section][name]['y']) != (len(mask), len(ids)):
                raise ValueError('Prediction bin/design axes differ from source identities')
    if set(raw['Design_indices']) & set(raw['Design_indices_validation']):
        raise ValueError('Training and holdout design identities overlap')
    return result, masks, wanted, len(reference)


def _selected_config(template, output, masks, covariance, overrides, analysis_name, parameterization):
    cfg = deepcopy(template)
    candidates = [name for name, value in cfg['analyses'].items()
                  if isinstance(value, dict) and 'emulators' in value.get('parameters', {})]
    if analysis_name is None:
        analysis_name = candidates[0] if len(candidates) == 1 else None
    if analysis_name not in candidates:
        raise ValueError('Specify an unambiguous native analysis_name')
    selected_analysis = cfg['analyses'][analysis_name]
    if parameterization is None:
        parameters = selected_analysis['parameterizations']
        parameterization = parameters[0] if len(parameters) == 1 else None
    if parameterization not in selected_analysis['parameterizations']:
        raise ValueError('Specify an unambiguous native parameterization')
    for label in (analysis_name, parameterization):
        if not label or Path(label).name != label or label in ('.', '..'):
            raise ValueError('Native output labels must be single path components')
    groups = selected_analysis['parameters']['emulators']
    selected = {}
    for name, group in groups.items():
        entries = [row for row in group['observable_list']
                   if any(key == (row['observable'] if isinstance(row, dict) else row)
                          or key.startswith((row['observable'] if isinstance(row, dict) else row) + '__')
                          for key in masks)]
        if not entries:
            continue
        group['observable_list'] = entries
        noise = group.get('simulation_noise', {})
        if (noise.get('residual_white_noise') is not False or noise.get('joint_groups') is not True
                or 'noise' in group['kernels']['active']):
            raise ValueError('Expected measured joint noise and no active WhiteKernel')
        group['simulation_noise']['covariance_file'] = str(covariance)
        selected[name] = group
    if not selected:
        raise ValueError('No native emulator groups cover the selected features')
    overrides = deepcopy(overrides or {})
    if set(overrides) - {'n_restarts', 'random_state', 'n_pc'}:
        raise ValueError('Only n_restarts, random_state and groupwise n_pc overrides are supported')
    for key, value in overrides.items():
        if key == 'n_pc' and not isinstance(value, dict):
            raise ValueError('n_pc overrides must be a group mapping')
        choices = value if isinstance(value, dict) else {name: value for name in selected}
        if set(choices) - set(selected):
            raise ValueError('Override names an unselected or unknown group')
        for name, item in choices.items():
            if type(item) is not int or item < (1 if key == 'n_pc' else 0):
                raise ValueError('Training overrides must be valid nonnegative integers (n_pc > 0)')
            if key == 'n_restarts':
                selected[name]['GPR'][key] = item
            else:
                selected[name][key] = item
    selected_analysis['parameters']['emulators'] = selected
    cfg['output_dir'] = str(output / 'results')
    cfg['observable_table_dir'] = str(output / 'inputs')
    cfg['observable_config_dir'] = str(output / 'inputs')
    cfg['observables_filename'] = 'observables.h5'
    return cfg, analysis_name, parameterization, overrides


def prepare(observables_path: Path, template_config: dict | Path | str,
            covariance_path: Path, selected_features: list[tuple[str, float, float]],
            output: Path, overrides: dict | None = None, *,
            analysis_name: str | None = None, parameterization: str | None = None) -> dict:
    """Stage selected raw bins and expose exact native pre-fit quantities.

    Input HDF5 is the accepted full raw table export, including validation rows.
    Bin selection precedes scaler/PCA fitting; no post-fit output masking occurs.
    ``output`` must not exist, even if empty. Preparation cannot be resumed by
    ``train``; use a different new output for each independent operation.
    """
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError('Training output must be a NEW directory: ' + str(output))
    source = Path(observables_path).resolve(strict=True)
    covariance = Path(covariance_path).resolve(strict=True)
    if source.suffix not in ('.h5', '.hdf5') or covariance.suffix != '.npz':
        raise ValueError('Training requires raw HDF5 observables and measured covariance NPZ')
    template_path = None if isinstance(template_config, dict) else Path(template_config).resolve(strict=True)
    template = template_config if template_path is None else yaml.safe_load(template_path.read_text())
    original = data_IO.read_dict_from_h5(source.parent, source.name, verbose=False)
    raw, masks, wanted, source_count = _filter_raw(original, selected_features)
    cfg, name, parameter, overrides = _selected_config(
        template, output, masks, covariance, overrides, analysis_name, parameterization)
    config_path = output / 'config.yaml'
    settings = analysis.AnalysisSettings.from_config(name, config_path, cfg, parameter)
    ec = interface.EmulationConfig.from_config_file(settings)
    choice = jg.options(ec)
    native_names = data_IO.sorted_observable_list_from_dict(raw, ec.observable_filter)
    features = _features(raw, native_names)
    grouped_features, covered = [], []
    for group, es in ec.emulation_settings.items():
        names = data_IO.sorted_observable_list_from_dict(raw, es.base_settings.observable_filter)
        rows = _features(raw, names)
        covered.extend(names)
        grouped_features.extend(rows)
        limit = min(len(rows), len(raw['Design']))
        if es.n_pc > limit or (es.max_n_components_to_calculate is not None
                              and es.n_pc > es.max_n_components_to_calculate):
            raise ValueError(f'{group}: configured n_pc unavailable; refusing to reduce it')
        if type(es.settings.get('random_state')) is not int:
            raise ValueError('Each native group requires an explicit integer random_state')
        if Path(base.IO.output_filename(es, settings).name).name != base.IO.output_filename(es, settings).name:
            raise ValueError('Invalid native model filename')
        if base.IO.output_filename(es, settings).parent != settings.output_dir:
            raise ValueError('Native model output escapes the local directory')
    if (len(covered) != len(set(covered)) or set(covered) != set(masks)
            or set(features) != set(wanted) or len(features) != len(wanted)
            or len(grouped_features) != len(features)):
        raise ValueError('Native filters must cover every selected observable exactly once')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'inputs').mkdir()
    with config_path.open('x') as stream:
        yaml.safe_dump(cfg, stream, sort_keys=False)
    data_IO.write_dict_to_h5(raw, settings.output_dir, 'observables.h5', verbose=False)
    models = {group: sk_learn.fit_emulator(es, settings, prepare_only=True)
              for group, es in ec.emulation_settings.items()}
    x = next(iter(models.values()))['_training_design']
    kernels, targets, projections, restart_blocks = [], [], [], []
    offset = 0
    for group, es in ec.emulation_settings.items():
        result = models[group]
        np.testing.assert_array_equal(result['_training_design'], x)
        pc, scaler = result['PCA']['pca'], result['PCA']['scaler']
        projection = pc.components_[:es.n_pc] / scaler.scale_
        if pc.whiten:
            projection /= np.sqrt(pc.explained_variance_[:es.n_pc, None])
        projections.append(projection)
        group_kernels = result['_initial_kernels']
        if any(sn.white_variance(kernel) != 0 for kernel in group_kernels):
            raise ValueError('Active WhiteKernel is unsupported in measured-noise REML training')
        kernels.extend(group_kernels)
        targets.append(result['PCA']['Y_pca_truncated'])
        bounds = np.vstack([kernel.bounds for kernel in group_kernels])
        rng = np.random.RandomState(es.settings['random_state'])
        restart_blocks.append([rng.uniform(bounds[:, 0], bounds[:, 1]) for _ in range(es.n_restarts)])
        result['joint_pc_gp'] = jg.GroupView(None, np.arange(offset, offset + es.n_pc))
        offset += es.n_pc
    first = next(iter(ec.emulation_settings.values()))
    noise = sn.load_aligned_covariance(covariance, raw['Design_indices'], grouped_features)
    projection = block_diag(*projections)
    noise_pc = np.einsum('ab,ibjc,dc->aidj', projection, noise, projection,
                         optimize=True).reshape(len(kernels) * len(x), -1)
    restarts = [np.concatenate([rows[i] for rows in restart_blocks]) for i in range(first.n_restarts)]
    native_order = [grouped_features.index(row) for row in features]
    native_noise = noise[:, native_order, :, :][:, :, :, native_order]
    forward, transfer, center, residual, groups = jg.linear_maps(ec, models)
    u, v = omitted_maps(ec, models, groups)
    truth = data_IO.predictions_matrix_from_h5(settings.output_dir, 'observables.h5',
                                             observable_filter=ec.observable_filter)
    fingerprint = sn.fingerprint(choice)
    report = dict(status='PREPARED', method='REML-constant-means-v1',
                  source_observables_sha256=_sha(source), covariance_sha256=fingerprint['sha256'],
                  template_config_sha256=_sha(template_path) if template_path else None,
                  source_feature_count=source_count, features=features,
                  requested_features=wanted, grouped_features=grouped_features,
                  training_ids=raw['Design_indices'].tolist(),
                  holdout_ids=raw['Design_indices_validation'].tolist(),
                  training_coordinates='native parameterization (no physical/unit conversion)',
                  overrides=overrides, n_pc={g: es.n_pc for g, es in ec.emulation_settings.items()},
                  seeds={g: es.settings['random_state'] for g, es in ec.emulation_settings.items()},
                  n_restarts=first.n_restarts, jitter=float(first.alpha),
                  initial_theta=np.concatenate([k.theta for k in kernels]).tolist(),
                  initial_bounds=np.vstack([k.bounds for k in kernels]).tolist(),
                  restart_thetas=[row.tolist() for row in restarts],
                  native_config=str(config_path), native_output=str(settings.output_dir))
    _json(output / 'preparation.json', report)
    return dict(features=features, grouped_features=grouped_features, raw=raw, models=models,
                ec=ec, settings=settings, config=cfg, report=report, x=x, truth=truth,
                targets=np.concatenate(targets, axis=1), initial_kernels=kernels,
                restart_thetas=restarts, projection=projection, noise_pc=noise_pc,
                noise=native_noise, grouped_noise=noise, fingerprint=fingerprint,
                maps=dict(projection=forward, transfer=transfer, center=center, residual=residual,
                          omitted_projection=u, omitted_transfer=v, group_indices=groups))


def _holdout_qa(bank, covariance):
    raw, model = bank['raw'], bank['model']
    if not len(raw['Design_indices_validation']):
        return None, dict(available=False, reason='No holdout rows')
    settings = bank['settings']
    points = data_IO.design_array_from_h5(settings.output_dir, 'observables.h5', validation_set=True)
    log_indices = data_IO.get_log_scale_indices(settings.raw_analysis_config, settings.parameterization)
    if log_indices:
        points = data_IO.apply_log_scale(points, log_indices)
    truth = data_IO.predictions_matrix_from_h5(settings.output_dir, 'observables.h5',
        validation_set=True, observable_filter=bank['ec'].observable_filter)
    latent = model.predict(points)
    reconstructed = (truth - model.center) @ model.projection.T @ model.transfer + model.center
    qa = dict(design_ids=raw['Design_indices_validation'], design_native=raw['Design_validation'],
              design_gp=points, truth=truth, latent_mean=latent['central_value'],
              latent_covariance=latent['cov'], pca_reconstruction=reconstructed,
              truncation_error=reconstructed - truth,
              interpolation_error=latent['central_value'] - reconstructed)
    ids = np.r_[raw['Design_indices'], raw['Design_indices_validation']]
    with np.load(covariance, allow_pickle=False) as source:
        available = set(ids) <= set(source['design_ids'])
    if available:
        noise = sn.load_aligned_covariance(covariance, ids, bank['features'])
        observed = model.predict(points, noise)
        qa.update(observed_mean=observed['central_value'], observed_covariance=observed['cov'])
    return qa, dict(available=True, paired_mc_available=bool(available))


def train(observables_path: Path, template_config: dict | Path | str,
          covariance_path: Path, selected_features: list[tuple[str, float, float]],
          output: Path, overrides: dict | None = None, *, optimize: bool = True,
          analysis_name: str | None = None, parameterization: str | None = None) -> dict:
    """Fresh table -> TRAINED mean-aware bank; optimization is on by default.

    ``optimize=False`` is an explicit small-test escape hatch, recorded as an
    unoptimized fit, not a collaborator baseline. Native caches are outputs only.
    ``bank['model'].predict(points, target_noise=None)`` returns full feature
    covariance; points must use the native GP parameterization.
    """
    if type(optimize) is not bool:
        raise ValueError('optimize must be an explicit boolean')
    bank = prepare(observables_path, template_config, covariance_path, selected_features,
                   output, overrides, analysis_name=analysis_name, parameterization=parameterization)
    first = next(iter(bank['ec'].emulation_settings.values()))
    started = time.monotonic()
    gp = _ProgressRestrictedMeanGP(bank['initial_kernels'], jitter=first.alpha,
                                  n_restarts=first.n_restarts, random_state=0)
    if optimize:
        gp._progress_started = gp._progress_last = started
        gp._progress_evaluations = 0
        gp._progress_seen = set()
        gp._progress_initials = [np.concatenate([k.theta for k in bank['initial_kernels']])
                                ] + bank['restart_thetas']
        print(f'REML_FIT_BEGIN designs={len(bank["x"])} features={len(bank["features"])} '
              f'pcs={len(bank["initial_kernels"])} starts={len(gp._progress_initials)}', flush=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        gp.fit(bank['x'], bank['targets'], bank['noise_pc'], optimize=optimize,
               restart_thetas=bank['restart_thetas'])
    theta = np.concatenate([kernel.theta for kernel in gp.kernels_])
    bounds = np.vstack([kernel.bounds for kernel in gp.kernels_])
    objective, gradient = gp.objective(theta)
    projected = gradient.copy()
    projected[(theta <= bounds[:, 0] + 1e-5) & (projected > 0)] = 0
    projected[(theta >= bounds[:, 1] - 1e-5) & (projected < 0)] = 0
    best = min(gp.optimization_, key=lambda row: row['objective']) if gp.optimization_ else None
    if optimize:
        print(f'REML_FIT_END objective={objective:.12g} '
              f'best_success={best["success"] if best else None} '
              f'seconds={time.monotonic() - started:.1f}', flush=True)
    report = bank['report']
    report.update(status='TRAINED' if optimize else 'TRAINED_UNOPTIMIZED', optimize=optimize,
                  seconds=time.monotonic() - started, optimization=gp.optimization_,
                  best_success=best['success'] if best else None,
                  warnings=[str(item.message) for item in caught], objective=float(objective),
                  gradient=gradient.tolist(), projected_gradient=projected.tolist(),
                  projected_gradient_max=float(abs(projected).max()) if projected.size else 0.,
                  fitted_theta=theta.tolist(), bounds=bounds.tolist(), joint_fit_id=uuid.uuid4().hex)
    if best is not None and not best['success']:
        report['status'] = 'OPTIMIZATION_FAILED'
        _json(bank['settings'].output_dir / 'joint_fit.json', report)
        raise RuntimeError('Best REML optimizer did not converge; diagnostics saved in joint_fit.json')
    for group, es in bank['ec'].emulation_settings.items():
        result = bank['models'][group]
        indices = result['joint_pc_gp'].indices
        result.update(joint_pc_gp=jg.GroupView(gp, indices),
                      emulators=[sn.PCView(gp, int(i)) for i in indices],
                      simulation_noise=bank['fingerprint'], joint_group_fit_id=report['joint_fit_id'],
                      joint_group_names=tuple(bank['ec'].emulation_settings),
                      mean_aware_training='REML-constant-means-v1')
        del result['_training_design'], result['_initial_kernels']
    maps = bank['maps']
    bank['model'] = FastMeanAware(gp, bank['truth'], maps['projection'], maps['transfer'],
        maps['center'], maps['residual'], maps['omitted_projection'], maps['omitted_transfer'], bank['noise'])
    bank['gp'] = gp
    qa, report['holdout_qa'] = _holdout_qa(bank, Path(covariance_path).resolve())
    for group, es in bank['ec'].emulation_settings.items():
        base.IO.write_emulator(bank['models'][group], es, bank['settings'])
    jg.validate_results(bank['ec'], bank['models'], check_file=True)
    output = Path(output)
    if qa is not None:
        np.savez_compressed(output / 'holdout_qa.npz', **qa)
    save_prediction_state(output / 'prediction_state.npz', bank['model'],
        features=bank['features'], design_ids=bank['raw']['Design_indices'], provenance=report)
    _json(bank['settings'].output_dir / 'joint_fit.json', report)
    _json(output / 'training_report.json', report)
    return bank

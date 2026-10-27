"""Pure selected-layout adapter for an already loaded nominal experiment.

No table/config/model loads, fits, predictions, plotting, or writes occur here.
Pass the native source-component and statistical covariance builders;
the returned experiment can be handed to native QA without changing covariance.
"""
from collections import defaultdict
from copy import deepcopy

import numpy as np


ATOL, RTOL = 1e-14, 1e-12


def _features(rows, label):
    result = []
    for row in rows:
        if len(row) != 3:
            raise ValueError(f"{label}: require exact (observable, lower, upper) triples")
        name, lower, upper = str(row[0]), float(row[1]), float(row[2])
        if not np.isfinite([lower, upper]).all() or lower >= upper:
            raise ValueError(f"{label}: invalid bin edges")
        result.append((name, lower, upper))
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{label}: empty or duplicate features")
    return result


def _matrix(value, size, label):
    result = np.asarray(value, dtype=float)
    if result.shape != (size, size) or not np.isfinite(result).all():
        raise ValueError(f"{label}: require a finite {size}-by-{size} matrix")
    np.testing.assert_allclose(result, result.T, atol=ATOL, rtol=RTOL, err_msg=label)
    return result


def _equal(actual, expected, label):
    np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL, err_msg=label)
    return dict(exact=bool(np.array_equal(actual, expected)),
                max_abs_error=float(np.max(np.abs(actual-expected), initial=0.)))


def _ranges(features):
    ranges, seen = [], set()
    for i, (name, _, _) in enumerate(features):
        if ranges and ranges[-1][2] == name:
            start, _, _ = ranges[-1]
            ranges[-1] = (start, i+1, name)
        else:
            if name in seen:
                raise ValueError("Each target observable must occupy one contiguous feature block")
            seen.add(name)
            ranges.append((i, i+1, name))
    return ranges


def select_experiment(full_exp, full_features, target_features, frozen_covariance, *,
                      group_matrices, build_statistical_covariance):
    """Return native exp reindexed in the exact requested target-feature order.

    ``group_matrices(exp)`` returns an ordered mapping of source tags to matrices;
    ``build_statistical_covariance(exp)`` returns the statistical matrix.
    No run/setup object is accepted or invoked. Metadata and all input arrays remain
    untouched. Source columns retain their original relative order and names.

    Applied correlation settings come from the caller and are preserved.
    Non-marginalizable conventions fail the component checks rather than being
    repaired. Numeric equality uses the module ATOL/RTOL constants; bitwise equality is
    reported separately under ``selected_experiment_qa``.
    """
    if "external_covariance" in full_exp:
        raise ValueError("External-total covariance mode cannot certify native source components")
    features = _features(full_features, "full layout")
    wanted = _features(target_features, "target layout")
    n = len(features)
    original = deepcopy(full_exp)
    manager = original["correlation_manager"]
    metadata = deepcopy(manager.to_dict())
    if list(original["observable_ranges"]) != _ranges(features):
        raise ValueError("Full observable ranges do not exactly partition the feature layout")
    if np.shape(original["y"]) != (n,) or np.shape(original["y_err_stat"]) != (n,):
        raise ValueError("Misaligned full experimental central values/statistical errors")
    names = list(original["systematic_names"])
    errors = np.asarray(original["y_err_syst"])
    if len(set(names)) != len(names) or errors.shape != (n, len(names)):
        raise ValueError("Misaligned or duplicate systematic columns")
    if any(not np.isfinite(np.asarray(original[key], dtype=float)).all()
           for key in ("y", "y_err_stat", "y_err_syst")):
        raise ValueError("Nonfinite full experimental arrays")
    lookup = {row: i for i, row in enumerate(features)}
    if any(row not in lookup for row in wanted):
        raise ValueError("Target contains a feature not exactly present in the full layout")
    indices = np.array([lookup[row] for row in wanted], dtype=int)
    ranges = _ranges(wanted)
    old_ranges = {name: (start, stop) for start, stop, name in original["observable_ranges"]}
    coordinates = {}
    for name, (start, stop) in old_ranges.items():
        x = np.asarray(metadata["_observable_pt"].get(name, []), dtype=float)
        if (x.shape != (stop-start,) or not np.isfinite(x).all()
                or np.any(x < -ATOL) or np.any(x > 1+ATOL)):
            raise ValueError(f"Missing or invalid full-measurement registered coordinates: {name}")
        coordinates[name] = x
    used_sources = {source for start, stop, name in original["observable_ranges"]
                    for source in manager.get_systematic_names_for_observable(name)}
    for source in used_sources:
        if source not in metadata["systematic_info"] or source not in names:
            raise ValueError("Manager/source-column membership mismatch: " + source)

    full_stat = _matrix(build_statistical_covariance(deepcopy(original)), n, "full statistical")
    full_sources = group_matrices(deepcopy(original))
    if list(full_sources) != list(manager.correlation_groups):
        raise ValueError("Source decomposition must retain native manager tag order")
    full_sources = {tag: _matrix(value, n, "full source " + str(tag))
                    for tag, value in full_sources.items()}
    full_systematic = manager.create_systematic_covariance_matrix(errors, names, n)
    _equal(sum(full_sources.values(), np.zeros((n, n))), full_systematic,
           "Full source decomposition differs from native manager")
    frozen = _matrix(frozen_covariance, n, "frozen experimental")
    full_check = _equal(full_stat + sum(full_sources.values(), np.zeros((n, n))), frozen,
                        "Full manager/components differ from frozen covariance")

    selected = {key: deepcopy(value) for key, value in original.items() if key != "correlation_manager"}
    selected["y"] = np.asarray(original["y"])[indices].copy()
    selected["y_err_stat"] = np.asarray(original["y_err_stat"])[indices].copy()
    selected["observable_ranges"] = ranges
    selected_registry = {name: manager.get_systematic_names_for_observable(name)
                         for _, _, name in ranges}
    active = {source for sources in selected_registry.values() for source in sources}
    selected_names = [source for source in names if source in active]
    columns = [names.index(source) for source in selected_names]
    selected["systematic_names"] = selected_names
    selected["y_err_syst"] = errors[np.ix_(indices, columns)].copy()
    selected_coordinates = {}
    for start, stop, name in ranges:
        old_start, _ = old_ranges[name]
        selected_coordinates[name] = coordinates[name][indices[start:stop]-old_start].copy()

    # Clone the applied source definitions instead of reparsing config strings,
    # which could rename auto-tags or resolve full-measurement lengths anew.
    metadata["observable_systematics"] = selected_registry
    metadata["all_systematic_names"] = selected_names
    metadata["systematic_info"] = {source: metadata["systematic_info"][source] for source in selected_names}
    metadata["correlation_groups"] = {}
    metadata["_observable_pt"] = selected_coordinates
    metadata["_persource_specs"] = {
        key: value for key, value in metadata.get("_persource_specs", {}).items()
        if any(key in name for name in selected_registry)}
    current = type(manager).from_dict(metadata)
    current.register_observable_ranges(ranges)
    current.set_observable_pt(selected_coordinates)
    # Native component addition is order-sensitive at floating-point precision.
    current.correlation_groups = defaultdict(list, (
        (tag, current.correlation_groups[tag]) for tag in manager.correlation_groups
        if tag in current.correlation_groups))
    selected["correlation_manager"] = current
    current_metadata = current.to_dict()
    if current_metadata["_pending_correlation_params"] != metadata["_pending_correlation_params"]:
        raise ValueError("Native correlation parameters changed during selection")
    for source in selected_names:
        if current_metadata["systematic_info"][source] != metadata["systematic_info"][source]:
            raise ValueError("Applied source definition changed during selection: " + source)
    for name, x in selected_coordinates.items():
        np.testing.assert_array_equal(current_metadata["_observable_pt"][name], x)

    external = {}
    old_by_range = {value: name for name, value in old_ranges.items()}
    new_by_name = {name: (start, stop) for start, stop, name in ranges}
    for key, info in original.get("per_observable_external_stat_cov", {}).items():
        old_range = (info["start"], info["end"])
        if old_range not in old_by_range:
            raise ValueError("External statistical block must match one full observable range")
        name = old_by_range[old_range]
        block = _matrix(info["matrix"], old_range[1]-old_range[0], "external statistical " + name)
        if name in new_by_name:
            start, stop = new_by_name[name]
            local = indices[start:stop]-old_range[0]
            external[key] = dict(deepcopy(info), start=start, end=stop,
                                 matrix=block[np.ix_(local, local)].copy())
    selected["per_observable_external_stat_cov"] = external

    size = len(wanted)
    principal = np.ix_(indices, indices)
    selected_stat = _matrix(build_statistical_covariance(deepcopy(selected)), size, "selected statistical")
    stat_check = _equal(selected_stat, full_stat[principal], "Statistical principal submatrix changed")
    selected_sources = group_matrices(deepcopy(selected))
    if list(selected_sources) != list(current.correlation_groups):
        raise ValueError("Selected source decomposition changed native tag order")
    source_checks = {}
    for tag, value in full_sources.items():
        expected = value[principal]
        actual = _matrix(selected_sources[tag], size, "selected source " + str(tag)) if tag in selected_sources else np.zeros((size, size))
        source_checks[tag] = dict(_equal(actual, expected, "Source principal submatrix changed: " + str(tag)),
                                  retained=tag in selected_sources)
    if set(selected_sources) - set(full_sources):
        raise ValueError("Selection invented a new systematic source group")
    systematic = sum(selected_sources.values(), np.zeros((size, size)))
    _equal(systematic, current.create_systematic_covariance_matrix(
        selected["y_err_syst"], selected_names, size), "Selected native systematic decomposition changed")
    total_check = _equal(selected_stat + systematic, frozen[principal],
                         "Selected total differs from frozen principal covariance")
    selected["selected_experiment_qa"] = dict(
        full_feature_indices=indices.tolist(), full_features=n, selected_features=size,
        full_nominal_reconstruction=full_check, statistical=stat_check,
        source_groups=source_checks, total=total_check,
        coordinates="Original full-measurement normalized centers, selected without renormalization",
        source_definitions_and_parameters_unchanged=True,
        covariance_policy="Exact principal marginal; no conditioning, repair, rescaling or refit",
        tolerances=dict(atol=ATOL, rtol=RTOL))
    return selected

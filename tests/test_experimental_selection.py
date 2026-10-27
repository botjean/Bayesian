"""Synthetic native-manager tests, without loading data, configs or models."""
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest


from bayesian.systematic_correlation import SystematicCorrelationManager
from bayesian.log_posterior import _build_statistical_covariance

from bayesian import experimental_selection as s


def group_matrices(exp):
    """Same isolation/tag-order semantics as the existing covariance checks."""
    result = {}
    manager = exp["correlation_manager"]
    for tag in manager.correlation_groups:
        isolated = exp["y_err_syst"].copy()
        for j, name in enumerate(exp["systematic_names"]):
            if name.rsplit(":", 1)[-1] != tag:
                isolated[:, j] = 0
        result[tag] = manager.create_systematic_covariance_matrix(isolated, exp["systematic_names"], len(exp["y"]))
    np.testing.assert_allclose(sum(result.values(), np.zeros((len(exp["y"]),)*2)),
        manager.create_systematic_covariance_matrix(exp["y_err_syst"], exp["systematic_names"], len(exp["y"])),
        atol=s.ATOL, rtol=s.RTOL)
    return result


def total(exp):
    return _build_statistical_covariance(exp) + sum(group_matrices(exp).values(), np.zeros((len(exp["y"]),)*2))


def select(exp, features, target, frozen):
    return s.select_experiment(exp, features, target, frozen,
        group_matrices=group_matrices, build_statistical_covariance=_build_statistical_covariance)


@pytest.fixture
def example():
    manager = SystematicCorrelationManager(correlation_groups=defaultdict(list))
    manager.parse_configuration([
        ("a", ["sys:shape_a", "sys,extra:shape_a", "sys,taa:norm"], [], None),
        ("b", ["sys:shape_b", "sys,luminosity:norm"], [], None),
        ("c", ["sys:shape_c"], [], None)])
    manager.set_correlation_parameters(dict(default="-1:1", shape_a="jetscape:0.2",
        shape_b="jetscape:0.2", shape_c="jetscape:0.2", norm="-1:0.7"))
    # Short registry keys deliberately differ from runtime labels.
    ranges = [(0, 5, "a__0-5"), (5, 9, "b__0-10"), (9, 11, "c__0-10")]
    manager.register_observable_ranges(ranges)
    manager.set_observable_pt({"a__0-5": np.array([0., .12, .35, .68, 1.]),
                               "b__0-10": np.array([0., .2, .7, 1.]),
                               "c__0-10": np.array([0., 1.])})
    names = manager.get_all_systematic_names()
    errors = np.zeros((11, len(names)))
    for start, stop, name in ranges:
        for j, source in enumerate(manager.get_systematic_names_for_observable(name)):
            errors[start:stop, names.index(source)] = np.linspace(.02, .08, stop-start)*(j+1)
    exp = dict(y=np.linspace(.1, 1.1, 11), y_err_stat=np.linspace(.02, .04, 11),
        y_err_syst=errors, systematic_names=names, correlation_manager=manager, observable_ranges=ranges,
        per_observable_external_stat_cov={
            "a__0-5": dict(start=0, end=5, matrix=.002*np.eye(5)+.0002*np.ones((5, 5)))})
    features = [(name, float(i), float(i+1)) for start, stop, name in ranges for i in range(stop-start)]
    return exp, features, total(exp)


def assert_same(left, right):
    if isinstance(left, dict):
        assert list(left) == list(right)
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    else:
        assert left == right


def test_noncontiguous_bins_external_stat_and_full_coordinate_scale(example):
    exp, features, frozen = example
    indices = np.array([1, 3, 5, 7])
    selected = select(exp, features, [features[i] for i in indices], frozen)
    np.testing.assert_array_equal(selected["y"], exp["y"][indices])
    np.testing.assert_array_equal(total(selected), frozen[np.ix_(indices, indices)])
    coordinates = selected["correlation_manager"].to_dict()["_observable_pt"]
    np.testing.assert_array_equal(coordinates["a__0-5"], [.12, .68])
    np.testing.assert_array_equal(coordinates["b__0-10"], [0., .7])
    stat = selected["per_observable_external_stat_cov"]["a__0-5"]
    assert (stat["start"], stat["end"]) == (0, 2)
    np.testing.assert_array_equal(stat["matrix"], exp["per_observable_external_stat_cov"]["a__0-5"]["matrix"][np.ix_([1, 3], [1, 3])])
    shapes = group_matrices(selected)["shape_a"]
    sigma = np.sqrt(np.sum(exp["y_err_syst"][[1, 3]][:, [exp["systematic_names"].index(x) for x in ("sys:shape_a", "sys,extra:shape_a")]]**2, axis=1))
    expected = np.outer(sigma, sigma)*np.exp(-np.square((np.array([.12, .68])[:, None]-[.12, .68])/.2))
    np.testing.assert_array_equal(shapes[:2, :2], expected)
    assert selected["selected_experiment_qa"]["total"]["exact"]


def test_drop_observable_preserves_cross_known_source(example):
    exp, features, frozen = example
    indices = [0, 2, 4, 6, 8]
    selected = select(exp, features, [features[i] for i in indices], frozen)
    manager = selected["correlation_manager"]
    assert list(manager.observable_systematics) == ["a__0-5", "b__0-10"]
    assert "c__0-10" not in manager.to_dict()["_observable_pt"]
    assert "sys:shape_c" not in selected["systematic_names"]
    assert "shape_c" not in manager.correlation_groups
    assert list(manager.correlation_groups) == [tag for tag in exp["correlation_manager"].correlation_groups if tag != "shape_c"]
    actual = group_matrices(selected)["norm"]
    expected = group_matrices(exp)["norm"][np.ix_(indices, indices)]
    np.testing.assert_array_equal(actual, expected)
    assert np.all(actual[:3, 3:] > 0)
    assert not selected["selected_experiment_qa"]["source_groups"]["shape_c"]["retained"]
    np.testing.assert_array_equal(total(selected), frozen[np.ix_(indices, indices)])


def test_reordered_observables_and_bins_are_exact_permutation(example):
    exp, features, frozen = example
    indices = [8, 5, 3, 1]
    selected = select(exp, features, [features[i] for i in indices], frozen)
    assert selected["observable_ranges"] == [(0, 2, "b__0-10"), (2, 4, "a__0-5")]
    assert list(selected["correlation_manager"].correlation_groups) == ["shape_a", "norm", "shape_b"]
    np.testing.assert_array_equal(total(selected), frozen[np.ix_(indices, indices)])
    assert selected["per_observable_external_stat_cov"]["a__0-5"]["start"] == 2


def test_input_and_returned_manager_arrays_do_not_alias(example, monkeypatch):
    exp, features, frozen = example
    before, metadata = deepcopy(exp), deepcopy(exp["correlation_manager"].to_dict())
    fcopy, ccopy = deepcopy(features), frozen.copy()

    def forbidden(*args, **kwargs):
        pytest.fail("Selected experiment adapter attempted file I/O")

    monkeypatch.setattr("builtins.open", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    selected = select(exp, features, features[:4]+features[6:8], frozen)
    assert selected["correlation_manager"] is not exp["correlation_manager"]
    for key in exp:
        if key != "correlation_manager":
            assert_same(exp[key], before[key])
    assert_same(exp["correlation_manager"].to_dict(), metadata)
    assert features == fcopy
    np.testing.assert_array_equal(frozen, ccopy)
    selected["y"][0] += 1
    selected["correlation_manager"].set_observable_pt({"a__0-5": np.zeros(4)})
    assert_same(exp["correlation_manager"].to_dict(), metadata)
    np.testing.assert_array_equal(exp["y"], before["y"])


@pytest.mark.parametrize("target", [[], [0, 0], [0, 5, 1]])
def test_invalid_empty_duplicate_or_interleaved_target_rejected(example, target):
    exp, features, frozen = example
    with pytest.raises(ValueError):
        select(exp, features, [features[i] for i in target], frozen)


def test_wrong_edges_and_frozen_covariance_fail_closed(example):
    exp, features, frozen = example
    wrong = [(features[0][0], features[0][1]+1e-12, features[0][2])]
    with pytest.raises(ValueError, match="not exactly present"):
        select(exp, features, wrong, frozen)
    with pytest.raises(AssertionError, match="frozen covariance"):
        select(exp, features, features[:3], frozen+np.eye(11)*.001)


@pytest.mark.parametrize('setting', ['jetscape:0.07', 'jetscape:0.63', 'ptgauss:0.41'])
def test_caller_correlation_settings_are_preserved(example, setting):
    exp, features, frozen = example
    legacy = deepcopy(exp)
    parameters = deepcopy(legacy["correlation_manager"].to_dict()["_pending_correlation_params"])
    parameters["shape_a"] = setting
    legacy["correlation_manager"].set_correlation_parameters(parameters)
    legacy["correlation_manager"].register_observable_ranges(legacy["observable_ranges"])
    selected = select(legacy, features, [features[0], features[3]], total(legacy))
    np.testing.assert_allclose(total(selected), total(legacy)[np.ix_([0, 3], [0, 3])], atol=1e-14)


def test_changed_bin_distance_fails_closed(example):
    exp, features, _ = example
    legacy = deepcopy(exp)
    parameters = deepcopy(legacy["correlation_manager"].to_dict()["_pending_correlation_params"])
    parameters["shape_a"] = "2:0.8"
    legacy["correlation_manager"].set_correlation_parameters(parameters)
    legacy["correlation_manager"].register_observable_ranges(legacy["observable_ranges"])
    with pytest.raises(AssertionError, match="Source principal submatrix changed"):
        select(legacy, features, [features[0], features[3]], total(legacy))


def test_identity_selection_is_native_reconstruction(example):
    exp, features, frozen = example
    selected = select(exp, features, features, frozen)
    np.testing.assert_array_equal(total(selected), frozen)
    np.testing.assert_array_equal(_build_statistical_covariance(selected), _build_statistical_covariance(exp))
    assert selected["selected_experiment_qa"]["total"] == dict(exact=True, max_abs_error=0.)

import numpy as np
import pytest

from bayesian.systematic_correlation import SystematicCorrelationManager, _parse_group_correlation_param
from bayesian.systematic_shift_shape import shift_shape_correlation


@pytest.mark.parametrize("n", [1, 2, 8, 20])
@pytest.mark.parametrize("t", [-1, -0.9, -0.5, 0, 0.5, 0.9, 1])
def test_modes_psd_and_cuts(n, t):
    correlation = shift_shape_correlation(n, t, n)
    np.testing.assert_array_equal(np.diag(correlation), 1)
    assert np.linalg.eigvalsh(correlation).min() > -1e-12
    selected = max(1, n // 2)
    np.testing.assert_allclose(
        shift_shape_correlation(selected, t, n), correlation[:selected, :selected]
    )


@pytest.mark.parametrize("value", ["shiftshape:2:3", "shiftshape:nan:3", "shiftshape:0:0",
                                         "shiftshape:0:3.5", "shiftshape:0", "shiftshape:0:3:4"])
def test_strict_parser(value):
    with pytest.raises(ValueError):
        _parse_group_correlation_param(value)


def manager(value):
    result = SystematicCorrelationManager()
    result.parse_configuration([("observable", ["sys:test"], [], None)])
    result.set_correlation_parameters({"test": value})
    result.resolve_bin_counts([(0, 3, "observable")])
    result.register_observable_ranges([(0, 3, "observable")])
    return result


@pytest.mark.parametrize("t", [-1, -0.5, 0, 0.5, 1])
def test_native_builder_and_serialization(t):
    result = manager(f"shiftshape:{t}:5")
    sigma = np.array([0.1, 0.2, 0.3])
    expected = np.outer(sigma, sigma) * shift_shape_correlation(3, t, 5)
    covariance = result.create_systematic_covariance_matrix(sigma[:, None], ["sys:test"], 3)
    np.testing.assert_allclose(covariance, expected)
    np.testing.assert_allclose(result.build_intra_observable_correlation_matrix("sys:test", 3),
                               shift_shape_correlation(3, t, 5))
    restored = SystematicCorrelationManager.from_dict(result.to_dict())
    actual = restored.create_systematic_covariance_matrix(sigma[:, None], ["sys:test"], 3)
    np.testing.assert_allclose(actual, expected)


def test_legacy_full_and_strength_unchanged():
    for value, rho in [("-1:1", 1), ("-1:0.5", 0.5), ("-1:0", 0)]:
        actual = manager(value).create_systematic_covariance_matrix(np.ones((3, 1)), ["sys:test"], 3)
        expected = np.full((3, 3), rho, dtype=float)
        np.fill_diagonal(expected, 1)
        np.testing.assert_array_equal(actual, expected)


def test_reject_shared_tag_and_undersized_original_count():
    result = manager("shiftshape:0:2")
    with pytest.raises(ValueError):
        result.create_systematic_covariance_matrix(np.ones((3, 1)), ["sys:test"], 3)
    result = SystematicCorrelationManager()
    result.parse_configuration([("first", ["sys:test"], [], None), ("second", ["sys:test"], [], None)])
    result.set_correlation_parameters({"test": "shiftshape:0:4"})
    result.resolve_bin_counts([(0, 2, "first"), (2, 4, "second")])
    result.register_observable_ranges([(0, 2, "first"), (2, 4, "second")])
    with pytest.raises(ValueError, match="multiple observables"):
        result.create_systematic_covariance_matrix(np.ones((4, 1)), ["sys:test"], 4)

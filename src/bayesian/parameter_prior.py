"""Evaluate configurable parameter priors in inference coordinates."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


def log_prior(
    samples: npt.NDArray[np.float64],
    parameterization_config: dict[str, Any],
) -> npt.NDArray[np.float64]:
    """Return the configured log-prior density in inference coordinates."""
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2:
        raise ValueError(
            "Expected samples with shape (n_samples, n_parameters), "
            f"got {samples.shape}"
        )

    prior_config = parameterization_config.get("prior")
    if prior_config is None:
        return np.zeros(samples.shape[0], dtype=np.float64)
    if prior_config.get("type") != "uniform_physical":
        raise ValueError(
            f"Unsupported prior type: {prior_config.get('type')!r}"
        )

    transform_config = parameterization_config.get(
        "physical_posterior_transform"
    )
    if transform_config is None:
        raise ValueError(
            "uniform_physical prior requires physical_posterior_transform"
        )
    components = transform_config.get("components", [])
    if len(components) != samples.shape[1]:
        raise ValueError(
            f"Transform has {len(components)} components, but samples have "
            f"{samples.shape[1]} parameters"
        )

    bounds = prior_config.get("bounds", {})
    component_names = {str(component["name"]) for component in components}
    unknown_bounds = set(bounds) - component_names
    if unknown_bounds:
        raise ValueError(
            "Prior bounds reference unknown transformed components: "
            f"{sorted(unknown_bounds)}"
        )

    output = np.zeros(samples.shape[0], dtype=np.float64)
    valid = np.ones(samples.shape[0], dtype=bool)
    for parameter_index, component in enumerate(components):
        name = str(component["name"])
        if name not in bounds:
            raise ValueError(
                f"uniform_physical prior is missing bounds for {name!r}"
            )
        lower, upper = (float(value) for value in bounds[name])
        if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
            raise ValueError(
                f"Invalid physical prior bounds for {name!r}: "
                f"[{lower}, {upper}]"
            )

        physical, derivative = _transform_with_derivative(
            samples[:, parameter_index],
            component.get("steps", []),
        )
        tolerance = (
            64
            * np.finfo(np.float64).eps
            * max(1.0, abs(lower), abs(upper))
        )
        component_valid = (
            np.isfinite(physical)
            & np.isfinite(derivative)
            & (physical >= lower - tolerance)
            & (physical <= upper + tolerance)
            & (np.abs(derivative) > 0)
        )
        valid &= component_valid
        with np.errstate(divide="ignore", invalid="ignore"):
            output[component_valid] += (
                np.log(np.abs(derivative[component_valid]))
                - np.log(upper - lower)
            )

    output[~valid] = -np.inf
    return output


def _transform_with_derivative(
    values: npt.NDArray[np.float64],
    steps: list[dict[str, Any]],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    transformed = np.asarray(values, dtype=np.float64).copy()
    derivative = np.ones_like(transformed)
    for step in steps:
        kind = step["kind"]
        if kind == "add":
            transformed += float(step["value"])
        elif kind == "subtract":
            transformed -= float(step["value"])
        elif kind == "multiply":
            factor = float(step["value"])
            transformed *= factor
            derivative *= factor
        elif kind == "divide":
            divisor = float(step["value"])
            transformed /= divisor
            derivative /= divisor
        elif kind == "power":
            power = float(step["value"])
            derivative *= power * np.power(transformed, power - 1)
            transformed = np.power(transformed, power)
        elif kind == "tan":
            derivative /= np.cos(transformed) ** 2
            transformed = np.tan(transformed)
        elif kind == "arctan":
            derivative /= 1 + transformed**2
            transformed = np.arctan(transformed)
        elif kind == "exp":
            transformed = np.exp(transformed)
            derivative *= transformed
        elif kind == "log":
            derivative /= transformed
            transformed = np.log(transformed)
        else:
            raise ValueError(f"Unknown transform step kind: {kind}")
    return transformed, derivative

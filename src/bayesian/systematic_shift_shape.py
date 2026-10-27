"""Error-preserving shift/contrast covariance diagnostic, explicitly opt-in."""

import numpy as np


def shift_shape_correlation(n_bins: int, parameter: float, original_bins: int) -> np.ndarray:
    """Return a principal block of the original standardized correlation model.

    The original bin count must be retained across cuts. Negative parameters
    suppress the common standardized mode, not the physical yield integral.
    """
    if not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    if not isinstance(original_bins, (int, np.integer)) or original_bins < n_bins:
        raise ValueError("original_bins must be an integer >= n_bins")
    if not np.isfinite(parameter) or not -1 <= parameter <= 1:
        raise ValueError("shift-shape parameter must be finite and in [-1, 1]")
    rho = parameter if parameter >= 0 else parameter / max(original_bins - 1, 1)
    result = np.full((n_bins, n_bins), rho, dtype=float)
    np.fill_diagonal(result, 1.0)
    return result

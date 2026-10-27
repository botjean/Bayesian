"""Define the likelihood separately for performance reasons

In doing so, we can use global variables. This isn't a nice thing to do from a coding perspective,
but it gives a significant improvement in MCMC performance during multiprocessing.
For the initial concept, see: https://emcee.readthedocs.io/en/stable/tutorials/parallel/#parallel

SYSTEMATIC UNCERTAINTY SUPPORT (August - November 2025, Jingyu Zhang)::
================================
Updated to use correlation-aware experimental data structure and systematic covariance matrices.

For detailed information on systematic correlation structure, see systematic_correlation.py
For covariance matrix visualization, see plot_covariance.py

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
.. codeauthor:: James Mulligan
.. codeauthor:: Jingyu Zhang <jingyu.zhang@cern.ch>, Vanderbilt
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.linalg import lapack

from bayesian import emulation, parameter_prior

logger = logging.getLogger(__name__)


g_min: npt.NDArray[np.float64] = None
g_max: npt.NDArray[np.float64] = None
# Parameter indices given a uniform-in-LOG prior (reweights the flat-in-linear prior by
# -sum(ln x_i)). Empty tuple = all parameters flat-in-linear (legacy behaviour).
g_log_prior_indices: tuple = ()
g_emulation_config: emulation.EmulationConfig = None
g_emulation_results: dict[str, dict[str, npt.NDArray[np.float64]]] = None
g_experimental_results: dict = None
g_emulator_cov_unexplained: dict = None
g_systematic_covariance: npt.NDArray[np.float64] = None
g_C_stat: npt.NDArray[np.float64] = None
g_parameterization_config: dict = None


def _build_systematic_covariance(experimental_results: dict) -> npt.NDArray[np.float64]:
    """
    Build the systematic covariance matrix from experimental_results, using the
    correlation_manager when available and otherwise falling back to a fully-correlated
    sum over each systematic source. Returns a (n_features, n_features) array.
    """
    n_features = len(experimental_results["y"])

    if "correlation_manager" in experimental_results:
        return experimental_results["correlation_manager"].create_systematic_covariance_matrix(
            experimental_results["y_err_syst"],
            experimental_results["systematic_names"],
            n_features,
        )

    # Fallback: no correlation manager. Treat each systematic source as fully correlated
    # within itself but uncorrelated with the others (the conservative legacy behaviour).
    cov = np.zeros((n_features, n_features))
    y_err_syst = experimental_results.get("y_err_syst")
    if y_err_syst is not None and y_err_syst.shape[1] > 0:
        for sys_idx in range(y_err_syst.shape[1]):
            sys_errors = y_err_syst[:, sys_idx]
            cov += np.outer(sys_errors, sys_errors)
    return cov


def _build_statistical_covariance(experimental_results: dict) -> npt.NDArray[np.float64]:
    """
    Build the statistical covariance: a diagonal of y_err_stat**2, with per-observable
    blocks replaced by the user-provided external statistical covariance where given.
    """
    n_features = len(experimental_results["y"])
    C_stat = np.zeros((n_features, n_features))
    np.fill_diagonal(C_stat, experimental_results["y_err_stat"] ** 2)

    per_obs = experimental_results.get("per_observable_external_stat_cov")
    if per_obs:
        for obs_label, cov_info in per_obs.items():
            start = cov_info["start"]
            end = cov_info["end"]
            C_stat[start:end, start:end] = cov_info["matrix"]
            logger.debug(f"External stat cov applied for {obs_label} (bins {start}-{end})")

    return C_stat


def save_covariance_matrices_for_plotting(experimental_results: dict, output_dir) -> None:
    """
    Compute the static covariance pieces (statistical + systematic) and pickle them to
    ``{output_dir}/covariance_matrices.pkl`` for later plotting. Call this once from the
    master process before creating a multiprocessing pool, so worker processes don't race
    on the same file.
    """
    if "external_covariance" in experimental_results:
        systematic_total = None
    else:
        systematic_total = _build_systematic_covariance(experimental_results)

    covariance_matrices = {
        "statistical": _build_statistical_covariance(experimental_results),
        "systematic_total": systematic_total,
        "emulator": None,  # Filled in by plotting code from emulator predictions.
    }

    output_file = Path(output_dir) / "covariance_matrices.pkl"
    with output_file.open("wb") as f:
        pickle.dump(covariance_matrices, f)
    logger.info(f"Saved covariance matrices to {output_file}")


def initialize_pool_variables(
    local_min,
    local_max,
    local_emulation_config,
    local_emulation_results,
    local_experimental_results,
    local_emulator_cov_unexplained,
    local_log_prior_indices=(),
) -> None:
    """
    Initialize globals for each multiprocessing-pool worker. The systematic and statistical
    covariance pieces are built here once per worker because they do not depend on the
    sampled parameters. Plot-side persistence (covariance_matrices.pkl) is handled by
    save_covariance_matrices_for_plotting on the master, not here.
    """

    global g_min, g_max, g_emulation_config, g_emulation_results, g_experimental_results
    global g_emulator_cov_unexplained, g_systematic_covariance, g_C_stat, g_log_prior_indices
    global g_parameterization_config
    g_min = local_min
    g_max = local_max
    g_log_prior_indices = tuple(local_log_prior_indices or ())
    g_emulation_config = local_emulation_config
    g_emulation_results = local_emulation_results
    g_experimental_results = local_experimental_results
    g_emulator_cov_unexplained = local_emulator_cov_unexplained
    parameterization = g_emulation_config.analysis_settings.parameterization
    g_parameterization_config = g_emulation_config.analysis_settings.raw_analysis_config[
        "parameterization"
    ][parameterization]

    if "external_covariance" in g_experimental_results:
        logger.info("External covariance mode: skipping systematic covariance construction")
        g_systematic_covariance = None

        ext_cov = g_experimental_results["external_covariance"]
        try:
            np.linalg.cholesky(ext_cov)
        except np.linalg.LinAlgError as e:
            raise ValueError(
                "External covariance matrix is not positive definite. "
                "Check the matrix you provided in experimental_results['external_covariance']."
            ) from e
        eigenvals = np.linalg.eigvalsh(ext_cov)
        logger.info(f"External covariance eigenvalues: min={eigenvals.min():.6e}, max={eigenvals.max():.6e}")
    else:
        if "correlation_manager" in g_experimental_results:
            logger.info("Calculating systematic covariance matrix for MCMC...")
        else:
            logger.info(
                "No correlation manager found - using fully-correlated-per-source fallback for systematic covariance"
            )
        g_systematic_covariance = _build_systematic_covariance(g_experimental_results)

    g_C_stat = _build_statistical_covariance(g_experimental_results)


# ---------------------------------------------------------------
def log_posterior(X, *, set_to_infinite_outside_bounds: bool = True) -> npt.NDArray[np.float64]:
    """
    Function to evaluate the log-posterior for a given set of input parameters.

    This function is called by https://emcee.readthedocs.io/en/stable/user/sampler/

    CHANGES August 2025
    - Updated data access to use 'y_err_stat' instead of 'y_err'
    - Added systematic covariance matrix to likelihood calculation
    - Maintains backward compatibility

    :param X input ndarray of parameter space values
    :param min list of minimum boundaries for each emulator parameter
    :param max list of maximum boundaries for each emulator parameter
    :param config emulation_configuration object
    :param emulation_results dict of emulation groups
    :param experimental_results arrays of experimental results
    """

    # Convert to 2darray of shape (n_samples, n_parameters)
    X = np.array(X, copy=False, ndmin=2)

    # Initialize log-posterior array, which we will populate and return
    log_posterior = np.zeros(X.shape[0])

    # Check if any samples are outside the parameter bounds, and set log-posterior to -inf for those
    inside = np.all((X > g_min) & (X < g_max), axis=1)  # noqa: SIM300
    # -1e300 is apparently preferred for pocoMC
    log_posterior[~inside] = -np.inf if set_to_infinite_outside_bounds else -1e300

    # Uniform-in-LOG prior on the configured (strictly positive) parameters: reweight the
    # flat-in-linear prior to log-uniform by adding -sum(ln x_i). Matches the paper's
    # log-uniform prior on c1/c2/c3. In-bounds samples have x > g_min > 0, so ln is safe.
    if g_log_prior_indices and np.any(inside):
        cols = list(g_log_prior_indices)
        log_posterior[inside] -= np.sum(np.log(X[inside][:, cols]), axis=1)

    # Evaluate log-posterior for samples inside parameter bounds
    n_samples = np.count_nonzero(inside)
    n_features = g_experimental_results["y"].shape[0]

    if n_samples > 0:
        log_posterior[inside] += parameter_prior.log_prior(
            X[inside], g_parameterization_config
        )
        # Get experimental data
        data_y = g_experimental_results["y"]

        # Compute emulator prediction
        # Returns dict of matrices of emulator predictions:
        #     emulator_predictions['central_value'] -- (n_samples, n_features)
        #     emulator_predictions['cov'] -- (n_samples, n_features, n_features)
        emulator_predictions = emulation.predict(
            X[inside],
            g_emulation_config,
            analysis_settings=g_emulation_config.analysis_settings,
            emulator_results=g_emulation_results,
            emulator_additional_covariance=g_emulator_cov_unexplained,
        )

        # Construct array to store the difference between emulator prediction and experimental data
        # (using broadcasting to subtract each data point from each emulator prediction)
        assert data_y.shape[0] == emulator_predictions["central_value"].shape[1]
        dY = emulator_predictions["central_value"] - data_y

        # Construct the covariance matrix: emulator + (external OR statistical + systematic).
        # The static pieces (g_C_stat, g_systematic_covariance) are precomputed once per
        # worker in initialize_pool_variables; we just broadcast them per sample here.
        covariance_matrix = np.zeros((n_samples, n_features, n_features))
        covariance_matrix += emulator_predictions["cov"]

        if "external_covariance" in g_experimental_results:
            covariance_matrix += g_experimental_results["external_covariance"][np.newaxis, :, :]
            if np.any(~np.isfinite(covariance_matrix)):
                logger.error("Non-finite values in covariance matrix!")
        else:
            covariance_matrix += g_C_stat[np.newaxis, :, :]
            if g_systematic_covariance is not None:
                covariance_matrix += g_systematic_covariance[np.newaxis, :, :]

        # Compute log likelihood at each point in the sample. Constant priors mean the
        # log-likelihood equals the log-posterior here (out-of-bounds samples were set to
        # -inf above).
        log_posterior[inside] += np.fromiter(
            map(_loglikelihood, dY, covariance_matrix),
            dtype=np.float64,
            count=n_samples,
        )

        # NOTE-STAT: We don't support the extra_std term here.

    return log_posterior


# ---------------------------------------------------------------
def _loglikelihood(y, cov):
    """
    Evaluate the multivariate-normal log-likelihood for difference vector `y`
    and covariance matrix `cov`:

        log_p = -1/2*[(y^T).(C^-1).y + log(det(C))] + const.

    The likelihood is NOT NORMALIZED, since this does not affect MCMC.
    The normalization const = -n/2*log(2*pi), where n is the dimensionality.

    Arguments `y` and `cov` MUST be np.arrays with dtype == float64 and shapes
    (n) and (n, n), respectively.  These requirements are NOT CHECKED.

    The calculation follows algorithm 2.1 in Rasmussen and Williams (Gaussian
    Processes for Machine Learning).

    """
    # Compute the Cholesky decomposition of the covariance.
    # Use bare LAPACK function to avoid scipy.linalg wrapper overhead.
    L, info = lapack.dpotrf(cov, clean=False)

    if info < 0:
        msg = "lapack dpotrf error: "
        msg += f"the {-info}-th argument had an illegal value"
        raise ValueError(msg)
    if info > 0:
        msg = "lapack dpotrf error: "
        msg += f"the leading minor of order {info} is not positive definite"
        raise np.linalg.LinAlgError(msg)

    # Solve for alpha = cov^-1.y using the Cholesky decomp.
    alpha, info = lapack.dpotrs(L, y)

    if info != 0:
        msg = "lapack dpotrs error: "
        msg += f"the {-info}-th argument had an illegal value"
        raise ValueError(msg)

    return -0.5 * np.dot(y, alpha) - np.log(L.diagonal()).sum()

    return -0.5 * np.dot(y, alpha) - np.log(L.diagonal()).sum()

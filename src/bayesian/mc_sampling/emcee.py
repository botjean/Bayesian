"""Sampling implementation using emcee

Canonical implementation of the mc_sampling plugin interface using the affine-invariant
ensemble sampler (emcee).

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import logging
import multiprocessing
import pickle
from pathlib import Path
from typing import Any, ClassVar

import attrs
import emcee
import numpy as np
import numpy.typing as npt

from bayesian import data_IO, emulation, log_posterior, parameter_prior
from bayesian.mc_sampling import base as mc_sampling_base

logger = logging.getLogger(__name__)

_register_name = "emcee"


def _apply_random_seed(sampler: emcee.EnsembleSampler, seed: int | None) -> None:
    """Seed emcee's internal RNG so runs are reproducible beyond initial positions."""
    if seed is None:
        return

    rng = np.random.RandomState(seed)
    sampler.random_state = rng.get_state()


def _draw_initial_positions(
    rng: np.random.Generator,
    parameter_min: npt.NDArray[np.float64],
    parameter_max: npt.NDArray[np.float64],
    n_walkers: int,
    parameterization_config: dict[str, Any],
) -> npt.NDArray[np.float64]:
    """Draw initial walkers within the configured prior support."""
    accepted: list[npt.NDArray[np.float64]] = []
    n_accepted = 0
    n_proposed = 0
    max_proposals = max(100_000, 1000 * n_walkers)
    while n_accepted < n_walkers and n_proposed < max_proposals:
        batch_size = max(32, 2 * (n_walkers - n_accepted))
        candidates = rng.uniform(
            parameter_min,
            parameter_max,
            (batch_size, len(parameter_min)),
        )
        selected = candidates[
            np.isfinite(
                parameter_prior.log_prior(
                    candidates,
                    parameterization_config,
                )
            )
        ]
        if selected.size:
            accepted.append(selected)
            n_accepted += len(selected)
        n_proposed += batch_size
    if n_accepted < n_walkers:
        raise RuntimeError(
            "Could not initialize enough walkers inside configured prior "
            f"support after {n_proposed} proposals"
        )
    return np.concatenate(accepted, axis=0)[:n_walkers]


@attrs.define
class SamplerSettings:
    """Settings for the emcee affine-invariant ensemble sampler.

    Attributes:
        sampler_name: Name matching _register_name.
        base_settings: Base sampler settings (holds raw config dict).
        n_walkers: Number of ensemble walkers.
        n_burn_steps: Total burn-in steps (split into two stages internally).
        n_sampling_steps: Number of production sampling steps.
        n_logging_steps: Log acceptance fraction every this many steps.
        settings: Raw MCMC config dict.
    """

    sampler_name: ClassVar[str] = "emcee"
    base_settings: mc_sampling_base.BaseSamplerSettings = attrs.field()
    n_walkers: int = attrs.field()
    n_burn_steps: int = attrs.field()
    n_sampling_steps: int = attrs.field()
    n_logging_steps: int = attrs.field()
    settings: dict[str, Any] = attrs.field()
    random_seed: int | None = attrs.field(default=None)
    # Ensemble move set. None/"stretch" -> emcee's default StretchMove (unchanged behaviour).
    # "de" -> differential-evolution mix (DEMove 0.8 + DESnookerMove 0.2), which proposes along
    # the ensemble's own covariance and mixes far better on strongly correlated posteriors
    # (here the alpha_s-Q0-tau0 ridge) where StretchMove stalls at low acceptance.
    moves: str | None = attrs.field(default=None)
    # DEMove mean stretch factor. None -> emcee default 2.38/sqrt(2*ndim) (~0.69 for 6 params).
    # Smaller values shorten DE proposals. Matters here because the unconstrained nuisances
    # (c1,c2,c3) span the whole prior box, so the ensemble spread -- and hence DE's difference
    # vectors -- is huge along those axes; full-length proposals exit the hard bounds and are
    # rejected (observed: acceptance ~0.036 with defaults, flat through production, unchanged
    # by the burn-in reseed since reseeding only contracts the CONSTRAINED directions).
    de_gamma0: float | None = attrs.field(default=None)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SamplerSettings:
        return cls(
            base_settings=mc_sampling_base.BaseSamplerSettings.from_config(config),
            n_walkers=config["n_walkers"],
            n_burn_steps=config["n_burn_steps"],
            n_sampling_steps=config["n_sampling_steps"],
            n_logging_steps=config["n_logging_steps"],
            random_seed=config.get("random_seed"),
            moves=config.get("moves"),
            de_gamma0=config.get("de_gamma0"),
            settings=config,
        )


def run_sampling(
    config: mc_sampling_base.MCConfig,
    emulation_config: emulation.EmulationConfig,
    emulation_results: dict[str, Any],
    experimental_results: dict[str, Any],
    parameter_min: npt.NDArray[np.float64],
    parameter_max: npt.NDArray[np.float64],
    parameter_ndim: int,
    parameter_log_prior_indices: tuple = (),
) -> None:
    """Run emcee-based MCMC.

    Markov chain Monte Carlo model calibration using the affine-invariant ensemble
    sampler (emcee). Uses a two-stage burn-in: walkers are reseeded at the most
    probable positions after the first half of burn-in to accelerate convergence.

    Args:
        config: MC sampling configuration (includes output paths, closure_index).
        emulation_config: Emulation configuration.
        emulation_results: Trained emulator results, keyed by emulator group name.
        experimental_results: Experimental data arrays.
        parameter_min: Lower bounds for each parameter.
        parameter_max: Upper bounds for each parameter.
        parameter_ndim: Number of parameters.
    """
    sampler_settings: SamplerSettings = config.sampler_settings  # type: ignore[assignment]

    # NOTE: We need to use `spawn` rather than `fork` on linux. Otherwise, some caching
    #       mechanisms (e.g. used in learning the emulator group mapping) don't work.
    # NOTE: We use `get_context` here to avoid having to globally specify the context.
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(
        initializer=log_posterior.initialize_pool_variables,
        initargs=[
            parameter_min,
            parameter_max,
            emulation_config,
            emulation_results,
            experimental_results,
            {},  # emulator_cov_unexplained: computed dynamically by predict() if needed
            parameter_log_prior_indices,
        ],
    ) as pool:
        logger.info("Initializing sampler...")
        # Move set: default (None / "stretch") keeps emcee's StretchMove so existing configs are
        # untouched. "de" uses the differential-evolution mix recommended by the emcee docs for
        # correlated targets; it fails loudly on an unknown name rather than silently falling back.
        move_name = (sampler_settings.moves or "stretch").lower()
        if move_name == "stretch":
            moves = None
        elif move_name == "de":
            # gamma0=None reproduces emcee's default, so `moves: de` without de_gamma0 is unchanged.
            moves = [(emcee.moves.DEMove(gamma0=sampler_settings.de_gamma0), 0.8),
                     (emcee.moves.DESnookerMove(), 0.2)]
            if sampler_settings.de_gamma0 is not None:
                logger.info(f"DEMove gamma0 = {sampler_settings.de_gamma0}")
        else:
            msg = f"Unknown mcmc 'moves' setting {sampler_settings.moves!r}; use 'stretch' or 'de'."
            raise ValueError(msg)
        logger.info(f"Ensemble moves: {move_name}")
        sampler = LoggingEnsembleSampler(
            sampler_settings.n_walkers,
            parameter_ndim,
            log_posterior.log_posterior,
            kwargs={"set_to_infinite_outside_bounds": True},
            pool=pool,
            moves=moves,
        )
        _apply_random_seed(sampler, sampler_settings.random_seed)

        rng = np.random.default_rng(sampler_settings.random_seed)
        parameterization_config = (
            emulation_config.analysis_settings.raw_analysis_config[
                "parameterization"
            ][emulation_config.analysis_settings.parameterization]
        )
        random_pos = _draw_initial_positions(
            rng,
            parameter_min,
            parameter_max,
            sampler_settings.n_walkers,
            parameterization_config,
        )

        # First half of burn-in from random positions
        logger.info(f"Parallelizing over {pool._processes} processes...")  # type: ignore[attr-defined]
        logger.info("Starting initial burn-in...")
        nburn0 = sampler_settings.n_burn_steps // 2
        sampler.run_mcmc(random_pos, nburn0, n_logging_steps=sampler_settings.n_logging_steps)

        # Reseed walkers at the most probable positions, then complete burn-in.
        # This significantly accelerates convergence and helps prevent stuck walkers.
        logger.info("Resampling walker positions...")
        X0 = sampler.flatchain[
            np.unique(sampler.flatlnprobability, return_index=True)[1][-sampler_settings.n_walkers :]
        ]
        sampler.reset()
        X0 = sampler.run_mcmc(
            X0, sampler_settings.n_burn_steps - nburn0, n_logging_steps=sampler_settings.n_logging_steps
        )[0]
        sampler.reset()
        logger.info("Burn-in complete.")

        # Production sampling
        logger.info("Starting production...")
        sampler.run_mcmc(X0, sampler_settings.n_sampling_steps, n_logging_steps=sampler_settings.n_logging_steps)

        # Collect results
        logger.info("Writing chain to file...")
        output_dict: dict[str, Any] = {
            "chain": sampler.get_chain(),
            "acceptance_fraction": sampler.acceptance_fraction,
            "log_prob": sampler.get_log_prob(),
            "random_seed": sampler_settings.random_seed if sampler_settings.random_seed is not None else -1,
            "chain_diagnostics": mc_sampling_base.compute_chain_diagnostics(
                sampler.get_chain(),
                np.asarray(parameter_min, dtype=np.float64),
                np.asarray(parameter_max, dtype=np.float64),
            ),
        }
        try:
            output_dict["autocorrelation_time"] = sampler.get_autocorr_time()
        except Exception as e:
            output_dict["autocorrelation_time"] = None
            logger.warning(f"Could not compute autocorrelation time:\n{e!s}")

        # For closure tests, save the design point parameters and pseudodata
        if config.closure_index >= 0:
            design_point = data_IO.design_array_from_h5(
                config.output_dir, filename="observables.h5", validation_set=True
            )[config.closure_index]
            output_dict["design_point"] = design_point

            cleaned_results: dict[str, Any] = {}
            for key in ["y", "y_err_stat"]:
                if key in experimental_results:
                    cleaned_results[key] = np.array(experimental_results[key], dtype=np.float64)
            if "y_err_syst" in experimental_results:
                cleaned_results["y_err_syst"] = np.array(experimental_results["y_err_syst"], dtype=np.float64)
            if "systematic_names" in experimental_results:
                cleaned_results["systematic_names"] = [str(name) for name in experimental_results["systematic_names"]]
            for key in ["y_err"]:
                if key in experimental_results and key not in cleaned_results:
                    cleaned_results[key] = experimental_results[key]
            experimental_results = cleaned_results

        config.mcmc_output_dir.mkdir(exist_ok=True, parents=True)
        data_IO.write_dict_to_h5(output_dict, config.mcmc_output_dir, "mcmc.h5", verbose=True)

        with Path(config.sampler_outputfile).open("wb") as f:
            pickle.dump(sampler, f)

        logger.info("Done.")


class LoggingEnsembleSampler(emcee.EnsembleSampler):  # type: ignore[misc]
    """emcee.EnsembleSampler with periodic acceptance-fraction logging."""

    def run_mcmc(
        self, X0: npt.NDArray[np.float64], n_sampling_steps: int, n_logging_steps: int = 100, **kwargs: Any
    ) -> Any:
        logger.info(f"  running {self.nwalkers} walkers for {n_sampling_steps} steps")
        result = None
        for n, result in enumerate(self.sample(X0, iterations=n_sampling_steps, **kwargs), start=1):
            if n % n_logging_steps == 0 or n == n_sampling_steps:
                af = self.acceptance_fraction
                logger.info(
                    f"  step {n}: acceptance fraction: mean {af.mean():.3f}, std {af.std():.3f}, "
                    f"min {af.min():.3f}, max {af.max():.3f}"
                )
        return result

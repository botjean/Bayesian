"""Defines the interface for interacting with emulators.

# Users interested in emulating expensive forward models

Emulation is handled through the `EmulationConfig` class. The concept is that
you can configure one or more emulators to provide emulation of the expensive
forward model. You use multiple emulators if you want:
- Different emulators for different observables. e.g. one devoted to hadron RAA,
  and another devoted to jet RAA.
- You want to use different packages to perform emulation.
You can mix and max these options as desired![^1]

Using the EmulationConfig, there are two main functionalities:
 - fit_emulator(), which trains the emulator(s) on the provided data.
 - predict() construct mean, std dev of emulator(s) for a given set of parameter values.

This code is based in part on JETSCAPE/STAT codebase.

[^1]: The impact of using different emulators for different observables at the same time
      hasn't been tested as of Nov 2025, so it should be used with care!

# Developers interested in implementing new emulation packages

If you're interested in implementing your own emulation package, you need to implement
the following functionality to interoperate with this package:

- fit_emulator(...): Function to train the emulator given some data:
  ```python
  def fit_emulator(
      emulator_settings: EmulatorSettings, analysis_settings: analysis.AnalysisSettings
  ) -> dict[str, Any]:
      '''Fit the emulator(s) to the data included in the analysis, and write to file.  # fmt: skip

    Args:
        emulator_settings: Overall emulation configuration.
        analysis_settings: Analysis settings.
    Returns:
        None.
    '''  # fmt: skip
      ...
  ```
- predict(...): Function to predict forward model values and covariance given a set of parameters.
  ```python
  def predict(
      parameters: npt.NDArray[np.float64],
      results: dict[str, Any],
      emulator_settings: EmulatorSettings,
      additional_covariance: npt.NDArray[np.float64] | None = None,
  ) -> dict[str, npt.NDArray[np.float64]]:
      '''Predict the values at the given parameters by calculating their expected value via the emulator.  # fmt: skip

      Args:
        parameters: Array of parameter values (e.g. [tau0, c1, c2, ...]), with shape (n_samples, n_parameters).
        results: Dictionary that stores output from the emulator.
        emulator_settings: Emulator settings.
        additional_covariance: Addition to the covariance due to the emulator.

      Returns:
          emulator_predictions: dictionary containing matrices of central values and covariance
      '''  # fmt: skip
      ...
  ```
- Additional covariance: Optional function to compute additional covariance terms, such as
  the additional covariance due to PCA truncation.
  ```python
  def compute_additional_covariance_contributions(
      emulator_settings: EmulatorSettings, emulator_result: dict[str, Any]
  ) -> npt.NDArray[np.float64]:
      '''Compute additional covariance contributions.  # fmt: skip

      Args:
          emulator_settings: Emulator settings.
          emulator_result: Emulator training results.
      Returns:
          Unexplained covariance
      '''  # fmt: skip
      ...
  ```
- EmulatorSettings: Class which provides the settings for the emulator. Must implement
  the emulation.base.EmulatorSettings Protocol! This settings class will be passed to
  the fit_emulator() and predict() functions. See the Protocol definition for further
  information.
- _register_name: Name under which the emulator will be registered. The framework
  will automatically register all modules in the `emulation` directory.

The `emulation.base` module contains functionality which helps implement emulators.
See it for further information.

The sk_learn module contains the canonical implementation as of Nov 2025, so if this
specification is unclear, it's often best to consult there.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import logging
from types import ModuleType
from typing import Any, TypeVar

import attrs
import numpy as np
import numpy.typing as npt

from bayesian import analysis, data_IO, register_modules
from bayesian.emulation import base as emulation_base

logger = logging.getLogger(__name__)

_emulators: dict[str, ModuleType] = {}


def _validate_emulator(name: str, module: ModuleType) -> None:
    """Validate that an emulator module follows the expected interface."""
    # Optional: This check is just for information!
    optional_functions = ["compute_additional_covariance_contributions"]
    found_optional_functions = []
    for function_name in optional_functions:
        if hasattr(module, function_name):
            found_optional_functions.append(function_name)

    if found_optional_functions:
        logger.info(f"Emulator module {name} implements the optional functions: {found_optional_functions}")
    else:
        logger.info(f"Emulator module {name} does not implement any optional functions.")


def fit_emulators(emulation_config: EmulationConfig, analysis_settings: analysis.AnalysisSettings) -> None:
    """Fit the emulator(s) to the data included in the analysis, and write to file.

    Args:
        emulation_config: Overall emulation configuration.
        analysis_settings: Analysis settings.
    Returns:
        None.
    """
    from bayesian.emulation import joint_groups
    if joint_groups.enabled(emulation_config):
        joint_groups.fit(emulation_config, analysis_settings)
        return
    # Fit the emulator for each emulation group
    emulators_output = {}

    for emulator_name, emulator_settings in emulation_config.emulation_settings.items():
        try:
            # The emulator name specifies the emulator package
            emulator = _emulators[emulator_settings.emulator_name]
        except KeyError as e:
            msg = f"Emulator backend '{emulator_settings.emulator_name}' not registered or available"
            raise KeyError(msg) from e

        logger.info(
            f"Fitting emulator for emulator '{emulator_name}' using backend '{emulator_settings.emulator_name}'"
        )

        emulators_output[emulator_name] = emulator.fit_emulator(
            emulator_settings=emulator_settings, analysis_settings=analysis_settings
        )
        # NOTE: Only write if it's not empty (e.g. if we've returned something meaningful).
        #       It may also return empty to signal that it's already trained, so we don't want overwrite
        #       that already trained emulator.
        if emulators_output[emulator_name]:
            emulation_base.IO.write_emulator(
                emulator_output=emulators_output[emulator_name],
                emulator_settings=emulator_settings,
                analysis_settings=analysis_settings,
            )
    # NOTE: We store everything in a dict so we can later return these if we decide it's helpful. However,
    #       it doesn't appear to be at the moment (August 2023), so we leave as is.


# def predict_from_emulator(
#     parameters: npt.NDArray[np.float64],
#     emulation_config: EmulationConfig,
#     merge_predictions_over_groups: bool = True,
#     emulation_group_results: dict[str, dict[str, Any]] | None = None,
#     emulator_cov_unexplained: dict[str, dict[str, Any]] | None = None,
# ) -> dict[str, npt.NDArray[np.float64]]:
#     # Called from MCMC
#     ...


def predict(
    parameters: npt.NDArray[np.float64],
    emulation_config: EmulationConfig,
    *,
    analysis_settings: analysis.AnalysisSettings,
    merge_predictions_over_groups: bool = True,
    emulator_results: dict[str, Any] | None = None,
    emulator_additional_covariance: dict[str, Any] | None = None,
) -> dict[str, npt.NDArray[np.float64]]:
    """Construct dictionary of emulator predictions for each observable

    Args:
        parameters: Array of parameter values (e.g. [tau0, c1, c2, ...]), with shape (n_samples, n_parameters)
        emulation_config: Configuration object for the overall emulator (including all groups).
        analysis_settings: Analysis settings.
        merge_predictions_over_groups: If True, merge predictions over emulators. If false, return a dictionary
            of predictions for each emulator. Default: True
        emulator_results: Dictionary containing results from each emulator. If None, read from file. Default: None.
        emulator_additional_covariance: Dictionary containing the additional covariance for each emulator. The source
            depends on the emulator (e.g. for PCA, this is the unexplained variance from the PCA). Generally we will
            precompute this in the MC sampling to save time, but if it is not precomputed (e.g. when plotting), we will
            automatically compute it here. If None, will be calculated. Default: None.
        emulator_predictions: Dictionary containing matrices of central values and covariance
    """
    from bayesian.emulation import joint_groups
    if joint_groups.enabled(emulation_config):
        return joint_groups.predict(parameters, emulation_config, analysis_settings,
                                    emulator_results, emulator_additional_covariance,
                                    merge_predictions_over_groups)
    if emulator_results is None:
        emulator_results = {}
    if emulator_additional_covariance is None:
        emulator_additional_covariance = {}

    predict_output = {}
    for emulator_name, emulator_settings in emulation_config.emulation_settings.items():
        emulator_result = emulator_results.get(emulator_name)
        # Only load the emulator directly from file if needed. If called frequently
        # (eg. in the MCMC), it's probably better to load it once and pass it in.
        # NOTE: I know that get() can provide a second argument as the default, but a quick check showed that
        #       `read_emulators` was executing far more than expected (maybe trying to determine some default value?).
        #       However, separating it out like this seems to avoid the issue, but better to just avoid the issue.
        if emulator_result is None:
            emulator_result = emulation_base.IO.read_emulator(
                emulator_settings=emulator_settings, analysis_settings=analysis_settings
            )

        # We need the emulator module to proceed further
        try:
            # The emulator name specifies the emulator package
            emulator = _emulators[emulator_settings.emulator_name]
        except KeyError as e:
            msg = f"Emulator backend '{emulator_settings.emulator_name}' not registered or available"
            raise KeyError(msg) from e

        # Compute additional covariance due to the emulator, if not precomputed. For example, this could
        # include the unexplained variance due to PC truncation for an emulator.
        if emulator_additional_covariance:
            additional_covariance = emulator_additional_covariance[emulator_name]
        elif hasattr(emulator, "compute_additional_covariance_contributions"):
            additional_covariance = emulator.compute_additional_covariance_contributions(
                emulator_settings=emulator_settings,
                emulator_result=emulator_result,
            )

        predict_output[emulator_name] = emulator.predict(
            parameters,
            emulator_result,
            emulator_settings,
            additional_covariance=additional_covariance,
        )

    # Allow the option to return immediately to allow the study of performance per emulation group
    if not merge_predictions_over_groups:
        return predict_output

    # Now, we want to merge predictions over groups
    return emulation_config.sort_observables_in_matrix.convert(group_matrices=predict_output)


@attrs.define
class EmulationConfig:
    """Emulation configuration.

    Emulation is handled by a group of one (or more) emulators. Each emulator can use a
    different emulator package, as well as a different selection of input data.
    """

    # analysis_config: dict[str, Any] = attrs.field(factory=dict)
    # emulator_settings: dict[str, emulation_base.EmulatorSettings] = attrs.field(factory=dict)
    # emulation_groups_config: dict[str, emulation_base.EmulatorSettings] = attrs.field(factory=dict)
    analysis_settings: analysis.AnalysisSettings = attrs.field()
    emulation_settings: dict[str, emulation_base.EmulatorSettings] = attrs.field(factory=dict)
    # config: dict[str, Any] = attrs.field(init=False)
    # Optional objects that may provide useful additional functionality
    _observable_filter: data_IO.ObservableFilter | None = attrs.field(init=False, default=None)
    _sort_observables_in_matrix: SortEmulationGroupObservables | None = attrs.field(init=False, default=None)

    @classmethod
    def from_config_file(
        # cls, analysis_name: str, parameterization: str, config_file: Path, analysis_config: dict[str, Any]
        cls,
        analysis_settings: analysis.AnalysisSettings,
    ):
        """
        Initialize the emulation configuration from a config file.
        """
        c = cls(analysis_settings=analysis_settings)
        # Initialize the config for each emulator
        c.emulation_settings = {}
        group_configs = analysis_settings.raw_analysis_config["parameters"][
            "emulators"
        ]
        resolved_filenames: dict[str, str] = {}
        for group_name, group_cfg in group_configs.items():
            emulator_settings = _emulators[group_cfg["emulator_package"]].EmulatorSettings.from_config(group_cfg)
            if "additional_name" in group_cfg:
                emulator_settings.additional_name = group_cfg["additional_name"]
            elif len(group_configs) > 1:
                emulator_settings.additional_name = group_name

            filename = emulation_base.IO.output_filename(
                emulator_settings=emulator_settings,
                analysis_settings=analysis_settings,
            ).name
            if filename in resolved_filenames:
                other_group = resolved_filenames[filename]
                msg = (
                    f"Emulator groups '{other_group}' and '{group_name}' both "
                    f"resolve to '{filename}'. Configure distinct "
                    "additional_name values."
                )
                raise ValueError(msg)
            resolved_filenames[filename] = group_name
            c.emulation_settings[group_name] = emulator_settings
        return c

    def read_all_emulator_groups(
        self, analysis_settings: analysis.AnalysisSettings
    ) -> dict[str, dict[str, npt.NDArray[np.float64]]]:
        """Read all emulator groups.

        Just a convenience function.
        """
        emulation_results = {}
        for emulator_name, emulator_settings in self.emulation_settings.items():
            emulation_results[emulator_name] = emulation_base.IO.read_emulator(
                emulator_settings=emulator_settings, analysis_settings=analysis_settings
            )
        return emulation_results

    @property
    def observable_filter(self) -> data_IO.ObservableFilter:
        if self._observable_filter is None:
            if not self.emulation_settings:
                msg = "Need to specify emulation groups to provide an observable filter"
                raise ValueError(msg)
            # Accumulate the include and exclude lists from all emulation groups
            include_list: list[str] = []
            exclude_list: list[str] = self.analysis_settings.raw_analysis_config.get(
                "global_observable_exclude_list", []
            )
            for emulator_config in self.emulation_settings.values():
                group_filter = emulator_config.base_settings.observable_filter
                if group_filter:
                    include_list.extend(group_filter.include_list)
                    exclude_list.extend(group_filter.exclude_list)

            self._observable_filter = data_IO.ObservableFilter(
                include_list=include_list,
                exclude_list=exclude_list,
            )
        return self._observable_filter

    @property
    def sort_observables_in_matrix(self) -> SortEmulationGroupObservables:
        if self._sort_observables_in_matrix is None:
            if not self.emulation_settings:
                msg = "Need to specify emulation groups to provide an sorting for observable group observables"
                raise ValueError(msg)
            # Accumulate the include and exclude lists from all emulation groups
            self._sort_observables_in_matrix = SortEmulationGroupObservables.learn_mapping(self)
        return self._sort_observables_in_matrix


@attrs.define
class SortEmulationGroupObservables:
    """Class to track and convert between emulation group matrices to match sorted observables.

    emulation_group_to_observable_matrix: Mapping from emulation group matrix to the matrix of observables. Format:
        {observable_name: (emulator_group_name, slice in output_matrix, slice in emulator_group_matrix)}
    shape: Shape of matrix output. Format: (n_design_points, n_features). Note that we may only be predicting
        one design point at a time, so we pick out the number of design points for the output based on the provided
        group outputs (which implicitly contains the required number of design points).
    available_value_types: Available value types in the group matrices. These will be extracted when the mapping is learned.
    """

    emulation_group_to_observable_matrix: dict[str, tuple[str, slice, slice]]
    shape: tuple[int, int]
    _available_value_types: set[str] | None = attrs.field(init=False, default=None)

    @classmethod
    def learn_mapping(cls, emulation_config: EmulationConfig) -> SortEmulationGroupObservables:
        """Construct this object by learning the mapping from the emulation group prediction matrices to the sorted and merged matrices.

        :param EmulationConfig emulation_config: Configuration for the emulator(s).
        :return: Constructed object.
        """
        # NOTE: This could be configurable (eg. for validation). However, we don't seem to immediately
        #       need this functionality, so we'll omit it for now.
        prediction_key = "Prediction"

        # Now we need the mapping from emulator groups to observables with the right indices.
        # First, we need to start with all available observables (beyond just what's in any given group)
        # to learn the entire mapping
        # NOTE: It doesn't matter what observables file we use here since it's just to find all of the observables which are used.
        all_observables = data_IO.read_dict_from_h5(emulation_config.analysis_settings.output_dir, "observables.h5")
        current_position = 0
        observable_slices = {}
        for observable_key in data_IO.sorted_observable_list_from_dict(all_observables[prediction_key]):
            n_bins = all_observables[prediction_key][observable_key]["y"].shape[0]
            observable_slices[observable_key] = slice(current_position, current_position + n_bins)
            current_position += n_bins

        # Now, take advantage of the ordering in the emulator groups. (ie. the ordering in the group
        # matrix is consistent with the order of the observable names).
        observable_emulation_group_map = {}
        for emulator_name, emulator_settings in emulation_config.emulation_settings.items():
            emulator_observable_keys = data_IO.sorted_observable_list_from_dict(
                all_observables[prediction_key], observable_filter=emulator_settings.base_settings.observable_filter
            )
            current_group_bin = 0
            for observable_key in emulator_observable_keys:
                observable_slice = observable_slices[observable_key]
                observable_emulation_group_map[observable_key] = (
                    emulator_name,
                    observable_slice,
                    slice(current_group_bin, current_group_bin + (observable_slice.stop - observable_slice.start)),
                )
                current_group_bin += observable_slice.stop - observable_slice.start
                logger.debug(
                    f"{observable_key=}, {observable_emulation_group_map[observable_key]=}, {current_group_bin=}"
                )
        logger.debug(f"Sorted order: {observable_slices=}")

        # And then finally put them in the proper sorted observable order
        observable_emulation_group_map = {k: observable_emulation_group_map[k] for k in observable_slices}

        # We want the shape to allow us to preallocate the array:
        # Default shape: (n_design_points, n_features)
        last_observable = list(observable_slices)[-1]
        shape = (all_observables[prediction_key][observable_key]["y"].shape[1], observable_slices[last_observable].stop)
        logger.debug(f"{shape=} (note: for all design points)")

        return cls(
            emulation_group_to_observable_matrix=observable_emulation_group_map,
            shape=shape,
        )

    def convert(
        self, group_matrices: dict[str, dict[str, npt.NDArray[np.float64]]]
    ) -> dict[str, npt.NDArray[np.float64]]:
        """Convert a matrix to match the sorted observables.

        Args:
            group_matrices: Matrixes to convert by emulation group. eg:
                {"group_1": {"central_value": np.array, "cov": [...]}, "group_2": np.array}.
        Returns:
            Converted matrix for each available value type.
        """
        if self._available_value_types is None:
            self._available_value_types = set([value_type for group in group_matrices.values() for value_type in group])  # noqa: C403

        output: dict[str, npt.NDArray[np.float64]] = {}
        # Requires special handling since we're adding matrices (ie. 3d rather than 2d)
        if "cov" in self._available_value_types:
            value_type = "cov"
            # FIX 2026-09-16: the previous implementation assembled the merged covariance as a block
            # diagonal of PER-OBSERVABLE blocks, which silently discarded every cross-observable element
            # of the emulator covariance -- including within a single emulation group, where the
            # PCA-based emulator covariance (GP part and truncated-PC part) is genuinely dense across
            # observables. STAT keeps the full matrix (src/emulator.py predict: gp_var . var_trans +
            # cov_trunc). Dropping it changed the likelihood substantially (chi2 307 vs 732 and
            # -log det 1916 vs 2192 at the same parameter points, 92 observables). Now every pair of
            # observables served by the SAME emulation group keeps its full block; pairs in different
            # groups stay zero (they are emulated independently).
            n_samples = next(iter(group_matrices.values()))[value_type].shape[0]
            n_features = self.shape[1]
            merged = np.zeros((n_samples, n_features, n_features))
            entries = list(self.emulation_group_to_observable_matrix.values())
            for group_i, out_i, grp_i in entries:
                group_cov = group_matrices[group_i][value_type]
                for group_j, out_j, grp_j in entries:
                    if group_j != group_i:
                        continue
                    merged[:, out_i, out_j] = group_cov[:, grp_i, grp_j]
            output[value_type] = merged

        # Handle the other values (as of 14 August 2023, it's just "central_value")
        for value_type in self._available_value_types:
            # Skip over "cov" since we handled it explicitly above.
            if value_type == "cov":
                continue

            # Since the number of design points that we want to predict varies, we can't define the output
            # until we can extract it from one group output. So we wait to initialize the output matrix until
            # we have the first group output.
            for observable_name, (  # noqa: B007
                emulation_group_name,
                slice_in_output_matrix,
                slice_in_emulation_group_matrix,
            ) in self.emulation_group_to_observable_matrix.items():
                emulation_group_matrix = group_matrices[emulation_group_name]
                if value_type not in output:
                    output[value_type] = np.zeros((emulation_group_matrix[value_type].shape[0], *self.shape[1:]))
                output[value_type][:, slice_in_output_matrix] = emulation_group_matrix[value_type][
                    :, slice_in_emulation_group_matrix
                ]

        return output


# Any float32 or float64 type, which we will use to pair input and output values.
T = TypeVar("T", np.float32, np.float64)


def nd_block_diag(arrays: list[npt.NDArray[T]]) -> npt.NDArray[T]:
    """Add 2D matrices into a block diagonal matrix in n-dimensions.

    See: https://stackoverflow.com/q/62384509

    Args:
        arrays: List of arrays to block diagonalize.
    Returns:
        Block diagonal matrix in n-dimensions.
    """
    shapes = np.array([i.shape for i in arrays])

    out = np.zeros(
        np.append(np.amax(shapes[:, :-2], axis=0), [shapes[:, -2].sum(), shapes[:, -1].sum()]), dtype=arrays[0].dtype
    )
    r, c = 0, 0
    for i, (rr, cc) in enumerate(shapes[:, -2:]):
        out[..., r : r + rr, c : c + cc] = arrays[i]
        r += rr
        c += cc

    return out


# def compute_emulator_cov_unexplained(
#     emulation_config: EmulationConfig, emulation_results, analysis_settings: analysis.AnalysisSettings
# ) -> dict:
#     """
#     Compute the predictive variance due to PC truncation, for all emulator groups.
#     See further details in compute_emulator_group_cov_unexplained().
#     """
#     emulator_cov_unexplained = {}
#     if not emulation_results:
#         emulation_results = emulation_config.read_all_emulator_groups(analysis_settings)
#     for emulator_name, emulator_settings in emulation_config.emulation_settings.items():
#         emulation_group_result = emulation_results.get(emulator_name)
#         emulator_cov_unexplained[emulator_name] = compute_emulator_group_cov_unexplained(
#             emulator_settings, emulation_group_result
#         )
#     return emulator_cov_unexplained


# Actually perform the discovery and registration of the emulators
if not _emulators:
    _emulators.update(
        register_modules.discover_and_register_modules(
            calling_module_name=__name__,
            # Explanation of required attributes:
            # Classes:
            #  - EmulatorSettings is the required class
            #    NOTE: We cannot trivially check that the EmulatorSettings class satisfies
            #          the protocol since we would have to instantiate the class, which
            #          isn't so trivial at this point. But we'll leave it as a separate block
            #          in case we get a better idea in the future.
            # Functions:
            #  - "fit_emulator"
            #  - "predict"
            #  - "compute_additional_covariance_contributions": Optional, so we
            #    check in the separate validation function.
            required_attributes=["EmulatorSettings", "fit_emulator", "predict"],
            validation_function=_validate_emulator,
        )
    )

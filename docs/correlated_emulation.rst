Correlated-noise, mean-aware emulation
======================================

The explicit opt-in APIs in ``bayesian.emulation`` support measured simulation
covariance, correlated conditioning across PCA groups, and marginalization of
unknown constant means. They do not change the default independent-GP path.

Training
--------

``training.prepare`` selects user-supplied observable/bin identities in an HDF5
table before fitting scalers and PCA. ``training.train`` additionally optimizes
signal kernels with ``mean_aware.RestrictedMeanGP`` and constructs the complete
observable-space predictor. No saved model or posterior is used as a training
input. The user supplies the native configuration, measured covariance archive,
selected feature triples and a new output directory. If the configuration
contains multiple analyses or parameterizations, select them explicitly.

Each native group retains its configured PC count, kernel family, bounds,
restart count and seed. The measured covariance is aligned by design ID and
observable/bin identity before joint projection. Cross-design, cross-bin and
cross-group noise terms are retained. No fitted white-noise kernel is added
by this REML entry point; numerical ``alpha`` is only the configured jitter.

The returned bank contains the fitted model, effective configuration,
training diagnostics and feature identities. ``bank['model'].predict(points)``
predicts latent simulator means and full feature covariance. Points use the
native configured emulator coordinates; applications supply their own physical
coordinate transformations and prior bounds.

Prediction and validation
-------------------------

``MeanAware`` and ``FastMeanAware`` use dense and reduced implementations of
the same unknown-mean Gaussian conditioning. ``omitted_coordinates.omitted_maps``
completes the omitted physical complement when a training SVD has fewer rows
than observable bins. This does not add fitted PCs or invent empirical physical
variation in absent SVD directions. Retained/omitted measured noise cross-blocks
are preserved.

Latent predictions do not acquire an independent copy of simulation noise.
Validation against simulated targets instead requires explicit test/train and
test/test covariance blocks, including any shared reference samples.
The native training helper reports paired holdout predictions when those
designs are present in the supplied covariance archive.

``review_state`` stores supported kernels and numerical conditioning arrays in
NPZ/JSON without loading executable pickle objects. It is a prediction-state
format, not a replacement for raw training tables or measured covariance.

Inference
---------

The general likelihood and sampling APIs remain ``bayesian.log_posterior`` and
``bayesian.mc_sampling``. The Gaussian likelihood includes the determinant of
the total experimental plus emulator covariance. ``parameter_prior`` handles
declared physical priors and transformation Jacobians. Applications choose
their data, coordinate maps, prior bounds, sampling or quadrature settings,
and reporting/plotting; none are prescribed here.

``experimental_selection.select_experiment`` preserves principal experimental
covariances, source sharing and original registered coordinates when selecting
bins, rather than renormalizing coordinates after a cut.

Limits
------

Measured covariance estimates, empirical PCA bases and fitted kernel parameters
are treated as given during prediction. Successful numerical tests do not
establish scientific coverage or compensate for missing experimental
correlations. See ``simulation_noise.rst`` for input formats and assumptions.
All implementation tests supplied here use synthetic data.

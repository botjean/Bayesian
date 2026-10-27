Known Simulation Noise (Opt-In)
==============================

The default PCA/independent-GP path is unchanged. To condition on supplied
simulation statistical covariance, add this to one emulator group's settings::

    simulation_noise:
      covariance_file: /absolute/path/to/simulation_covariance.npz
      residual_white_noise: false

Remove ``noise`` from ``kernels.active`` for this configuration. To test an
additional fitted residual nugget, set ``residual_white_noise: true`` and retain
the existing ``noise`` kernel configuration. Inconsistent settings fail rather
than silently including or dropping a WhiteKernel. Use a separate output
directory or explicitly retrain; incompatible cached fits are rejected.

Input format
------------

The NPZ file contains ``design_ids`` (N integers), ``observable_names`` (F
strings), ``xmin`` and ``xmax`` (F bin edges), and ``covariance`` with shape
(N,F,N,F). Values are covariances in original observable units. Observable names
and bin edges identify each feature. IDs identify design rows. The training
loader aligns both against the selected HDF5 inputs, including pT cuts, and
excludes held-out designs. Duplicate/missing identities, nonfinite values,
asymmetry, and non-positive-semidefinite covariance are rejected.

Independent per-bin errors may be represented by a diagonal matrix, but that
is an assumption the producer must make explicitly. Ratios using the same pp
sample generally require nonzero cross-design blocks. Covariance-source hashes
are stored; reusing a fit through the training entry point checks the source
hash. Prediction uses the covariance frozen into the fit, without rereading
the external file at every likelihood call.

Correlations between emulator groups
------------------------------------

Set ``joint_groups: true`` inside ``simulation_noise`` for every selected group
to retain cross-group simulation covariance. All groups must use the same NPZ
bank and residual-noise mode, with matching numerical jitter and restart counts.
The separate PCA bases, whitening settings, numbers of PCs and kernel families
remain unchanged. Their retained scores are conditioned together, not fitted
independently and joined as a block-diagonal prediction afterwards. Each group's
configured seed supplies its block of each joint optimizer restart.

The native fit/predict entry points handle this mode; directly fitting a single
group is rejected. Group pickles retain a common fit identifier and membership
list. Mixing fits or changing group membership is rejected. Standard single-group
QA receives the appropriate marginal of the same joint conditional distribution.
The merged predictor retains cross-group observable covariance. With the option
absent, the earlier single-group known-noise behavior remains unchanged.

Model
-----

The linear projection is A = PC_components / StandardScaler.scale_. Whitened
PCs additionally divide each projection row by sqrt(explained_variance).
The full noise covariance becomes (I x A) C_MC (I x A)^T, reordered PC-major.
No cross-design or cross-PC terms are discarded. The empirical PCA basis and
centering are treated as fixed, as in the existing implementation.

Each retained PC keeps its own independent sklearn GP prior kernel. Known
correlated observation noise couples their conditioning and hyperparameter
optimization, so a joint dense Gaussian solve is required. All kernel
hyperparameters are fit by joint marginal likelihood with the configured
restart count/seed. GPR.alpha remains numerical diagonal jitter, not a stand-in
for the MC uncertainty. PC cross-covariances induced by conditioning are
retained in the observable-space emulator covariance.

Predictions target the latent simulator mean. Supplied MC noise and an optional
fitted residual WhiteKernel enter the training covariance, but neither is added
as fresh independent noise to latent predictions. Their effect on uncertainty
through conditioning remains. This differs deliberately from legacy sklearn
WhiteKernel prediction, which includes the nugget in prediction variance.
A residual nugget can absorb underestimated MC errors or unresolved simulation
variation; it is not automatically needed and does not describe model-data
discrepancy. Fitted residual noise at its lower bound is acceptable when known
noise already explains the training scatter.

Limits and validation
---------------------

The PCA-truncation covariance is unchanged: omitted PCs retain their empirical
total variance, which can include simulation noise. This option does not fit a
noise-aware PCA, infer uncertain MC covariance estimates, or correct the input
error convention (per-jet versus event-level, normalization errors, etc.).
It is a dense method intended for modest training banks, not thousands of
designs and retained PCs. These assumptions must be reported with comparisons.

Native QA plots show latent predictions versus noisy design values; their
normalized-residual widths alone are not a calibrated noisy-target coverage
test. For held-out coverage with shared MC samples, JointPCGP.predict accepts
the test noise and test/training cross-noise blocks. Conditioning then predicts
the noisy held-out targets, including the optional residual nugget at the test
point. Fivefold PCA/scaling must be fit on each training fold only.

Tests compare joint conditioning to explicit matrix algebra and sklearn's
diagonal-alpha case, check analytic gradients, whitening, ID/bin alignment,
cross-PC covariance, batch invariance, pickle round-trips, cache mismatches,
and the unchanged legacy path.

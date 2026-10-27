Shift/contrast correlation diagnostic
====================================

This optional sensitivity model does not reconstruct experimental covariance.
Legacy correlation settings are unchanged. Enable it only for explicitly chosen
unresolved source tags, each belonging to one observable::

    analyses:
      correlation_groups:
        default: '-1:1'
        unresolved_shape: 'shiftshape:0.5:15'

The fields are the parameter ``t`` in [-1,1] and the original bin count before
selection cuts. Marginal errors are preserved. Off-diagonal correlations equal
``t`` for nonnegative values and ``t/(original_bins-1)`` for negative values.
``t=1`` is coherent, ``t=0`` diagonal, and ``t=-1`` suppresses the common
standardized mode in the full original distribution. That last constraint is
not a zero-integrated-yield constraint. Cutting bins selects a principal block;
it must not change the original count.

Documented normalizations, detector-coherent terms, statistical and simulation
covariances must be kept separate. A shared parameter does not establish shared
nuisances across observables. Multiple-observable tags are rejected rather than
silently assigning new cross-observable correlations. Retagging unresolved
sources independently changes their old cross-observable assumption and must
be disclosed and compared separately. Singular source endpoints are allowed;
the complete likelihood covariance still must be positive definite.

Correlation-manager metadata is stored in ``observables.h5``. For a normal
steer_analysis run, reinitialize the observable cache when changing its source
tags or correlation settings; editing YAML with ``initialize_observables: false``
does not update an existing manager automatically. Covariance-only workflows can
instead construct/update the experimental manager in memory and call the native
builder explicitly, leaving the frozen data/prediction cache and GP fits alone.

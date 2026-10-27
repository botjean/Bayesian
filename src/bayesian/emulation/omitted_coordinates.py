"""Complete omitted coordinates only; leave native retained PCA and R untouched."""
import numpy as np
from scipy.linalg import null_space


def omitted_maps(config, models, group_indices):
    f = config.sort_observables_in_matrix.shape[1]
    projections, transfers = [], []
    for name, settings in config.emulation_settings.items():
        cached = models[name]['PCA']
        pca, scaler = cached['pca'], cached['scaler']
        idx = group_indices[name]
        components = np.asarray(pca.components_)
        retained = settings.n_pc
        assert components.shape[1] == len(idx)
        assert 0 < retained <= len(components) <= len(idx)
        assert np.isfinite(scaler.scale_).all() and (scaler.scale_ > 0).all()
        np.testing.assert_allclose(components @ components.T, np.eye(len(components)), atol=1e-10)
        if len(components) == len(idx):
            # Exactly the frozen code path, including operation order/whitening.
            omitted = components[retained:]
            factors = (np.sqrt(pca.explained_variance_[retained:]) if pca.whiten
                       else np.ones(len(omitted)))
            if np.any(factors <= 0):
                raise ValueError('Degenerate whitened omitted component')
        else:
            # SVD returns min(n_designs, n_features) rows. Its absent directions
            # still carry measured MC noise and unknown-mean uncertainty. Use
            # all f-p orthonormal omitted directions, with no whitening of zeros.
            omitted = null_space(components[:retained]).T
            assert omitted.shape == (len(idx) - retained, len(idx))
            factors = np.ones(len(omitted))
        forward, inverse = np.zeros((len(omitted), f)), np.zeros((len(omitted), f))
        forward[:, idx] = omitted / scaler.scale_ / factors[:, None]
        inverse[:, idx] = omitted * scaler.scale_ * factors[:, None]
        projections.append(forward)
        transfers.append(inverse)
    return np.vstack(projections), np.vstack(transfers)

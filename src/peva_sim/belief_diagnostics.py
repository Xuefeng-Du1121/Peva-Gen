"""Evaluator-only diagnostics of simulator particles, not learned PEVA beliefs."""
import numpy as np


def particle_diagnostics(particles, weights, targets, found):
    particles, weights, targets = [np.asarray(x, dtype=np.float64)
                                  for x in (particles, weights, targets)]
    found = np.asarray(found, dtype=bool)
    if (particles.shape != weights.shape + (2,) or weights.ndim != 2
            or targets.shape != (len(weights), 2)
            or found.shape != (len(weights),)):
        raise ValueError('Incompatible posterior shapes')
    if not all(np.isfinite(x).all() for x in (particles, weights, targets)) or (weights < 0).any():
        raise ValueError('Invalid posterior values')
    mass = weights.sum(axis=1)
    scale = weights.max(axis=1)
    valid = ~found & (scale > 0)
    mean = np.full(targets.shape, np.nan)
    error = np.full(len(weights), np.nan)
    ess = np.full(len(weights), np.nan)
    # Rescale before normalizing to support subnormal intensity masses.
    normalized = weights[valid] / scale[valid, None]
    normalized /= normalized.sum(axis=1, keepdims=True)
    mean[valid] = (particles[valid] * normalized[..., None]).sum(axis=1)
    error[valid] = np.linalg.norm(mean[valid] - targets[valid], axis=-1)
    ess[valid] = 1. / np.square(normalized).sum(axis=1)
    return mean, error, ess, mass, valid


def summarize_particles(errors, ess, mass, n_particles):
    errors, ess, mass = map(np.asarray, (errors, ess, mass))
    valid = np.isfinite(errors) & np.isfinite(ess)
    def average(values):
        return float(np.mean(values[valid])) if valid.any() else None
    return {
        'source': 'simulator_particle_posterior',
        'scope': 'pre-action undetected targets with positive particle mass',
        'valid_target_steps': int(valid.sum()),
        'total_target_steps': int(valid.size),
        'posterior_mean_error_m': average(errors),
        'posterior_mean_error_p95_m': float(np.percentile(errors[valid], 95)) if valid.any() else None,
        'effective_sample_size_mean': average(ess),
        'effective_sample_size_fraction': average(ess / n_particles),
        'remaining_mass_mean': average(mass),
    }

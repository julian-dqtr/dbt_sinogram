"""Statistics of evaluate_all.py behind the reach and oversmoothing figures (no model, no GPU)."""
import numpy as np
import pytest

from src_2D.utils.metrics import PerViewMoments, smoothness_metrics


def accumulate(samples) -> PerViewMoments:
    moments = PerViewMoments()
    for sample in samples:
        moments.update(sample)
    return moments


def test_per_view_variance_is_the_population_variance_averaged_over_the_detector():
    rng = np.random.default_rng(0)
    samples = 3.0 + rng.normal(size=(50, 7, 5)) * rng.uniform(0.1, 2.0, size=(1, 7, 5))
    np.testing.assert_allclose(accumulate(samples).per_view_variance(), samples.var(axis=0).mean(axis=1), rtol=1e-10)


def test_identical_outputs_have_exactly_zero_variance():
    """A graph model beyond its reach outputs the same values for every sample: the variance must
    be exactly 0, not a rounding residual, since the notebook checks it against 0."""
    constant = 0.1 + 1e3 * np.random.default_rng(1).normal(size=(7, 5))
    assert np.all(accumulate([constant] * 200).per_view_variance() == 0.0)


def test_ground_truth_variance_is_the_floor_of_any_prediction_independent_of_the_data():
    rng = np.random.default_rng(2)
    targets = rng.normal(size=(30, 7, 5))
    floor = accumulate(targets).per_view_variance()
    for constant in (np.zeros((7, 5)), rng.normal(size=(7, 5)), targets.mean(axis=0)):
        per_view_mse = ((targets - constant) ** 2).mean(axis=(0, 2))
        assert np.all(per_view_mse >= floor - 1e-12)
    # The bound is reached by the mean of the samples.
    np.testing.assert_allclose(((targets - targets.mean(axis=0)) ** 2).mean(axis=(0, 2)), floor)


# Two blocks of missing views around an acquired window, as in the real geometry.
MISSING = np.array([True] * 4 + [False] * 3 + [True] * 4)


def test_smoothness_ratios_are_one_for_the_ground_truth():
    target = np.random.default_rng(3).normal(size=(11, 16))
    assert smoothness_metrics(target, target, MISSING) == pytest.approx(
        {"dirichlet_angle_ratio": 1.0, "dirichlet_detector_ratio": 1.0})


def test_a_washed_out_wedge_has_no_angular_energy():
    rng = np.random.default_rng(4)
    target = rng.normal(size=(11, 16))
    washed_out = target.copy()
    washed_out[MISSING] = target[MISSING].mean(axis=0)  # the same row in every missing view
    metrics = smoothness_metrics(washed_out, target, MISSING)
    assert metrics["dirichlet_angle_ratio"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["dirichlet_detector_ratio"] < 1.0


def test_smoothness_ignores_the_acquired_views():
    """Acquired views are copied by data consistency: they, and the pairs of views that straddle
    the border of the window, must not enter the wedge indicators."""
    rng = np.random.default_rng(5)
    target, pred = rng.normal(size=(11, 16)), rng.normal(size=(11, 16))
    altered = pred.copy()
    altered[~MISSING] = 1e3 * rng.normal(size=((~MISSING).sum(), 16))
    assert smoothness_metrics(altered, target, MISSING) == pytest.approx(smoothness_metrics(pred, target, MISSING))

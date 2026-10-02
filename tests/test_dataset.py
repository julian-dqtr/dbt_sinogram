"""Seeded, immutable and disjoint splits that never touch the global RNG state."""
from typing import Tuple, cast

import numpy as np
import torch

from src_2D.data.dataset_2d import SinogramCompletionDataset


def _equal(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b))


def _numpy_key_state() -> np.ndarray:
    """The Mersenne Twister key of the global numpy RNG (legacy tuple layout)."""
    return np.asarray(cast(Tuple, np.random.get_state(legacy=True))[1])


def test_samples_are_reproducible_including_noise(config):
    first = SinogramCompletionDataset(4, split="test", geometry_config=config)
    second = SinogramCompletionDataset(4, split="test", geometry_config=config)  # fresh instance
    np.random.seed(123); torch.manual_seed(123)
    a = first[2]
    np.random.seed(999); torch.manual_seed(999)  # a different global state must not matter
    assert _equal(a, first[2]) and _equal(a, second[2])
    assert not torch.equal(a[0][a[0] != 0], a[1][a[0] != 0]), "the measured views should be noisy"


def test_global_rng_state_is_untouched(config):
    dataset = SinogramCompletionDataset(2, split="val", geometry_config=config)
    torch_state, numpy_state = torch.get_rng_state().clone(), _numpy_key_state().copy()
    _ = dataset[0], dataset[1]
    assert torch.equal(torch_state, torch.get_rng_state())
    assert np.array_equal(numpy_state, _numpy_key_state())


def test_splits_are_disjoint(config):
    phantoms = {split: torch.stack([SinogramCompletionDataset(6, split=split, noise_level=0.0, geometry_config=config)[i][2]
                                    for i in range(6)]) for split in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        for i in range(6):
            for j in range(6):
                assert not torch.equal(phantoms[a][i], phantoms[b][j]), f"{a}[{i}] == {b}[{j}]"


def test_fixed_vs_resampled_training_set(config):
    fixed = SinogramCompletionDataset(2, split="train", geometry_config=config)
    resampled = SinogramCompletionDataset(2, split="train", fixed_train_set=False, geometry_config=config)
    assert _equal(fixed[0], fixed[0])
    assert not torch.equal(resampled[0][2], resampled[0][2])


def test_target_is_clean_and_input_is_cropped(config, geom):
    noisy = SinogramCompletionDataset(1, split="val", geometry_config=config)[0]
    clean = SinogramCompletionDataset(1, split="val", noise_level=0.0, geometry_config=config)[0]
    assert torch.equal(noisy[1], clean[1]), "the target must not carry noise"
    missing = torch.from_numpy(~geom.acquired_view_mask)
    assert torch.all(noisy[0][0][missing] == 0) and noisy[0].shape == (1, 180, 128)

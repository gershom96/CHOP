import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from datasets.policy_image_cache import policy_pixels
from training.policy_reward_loading import ResumableOrder, legacy_first_epoch_order


def test_ssd_cache_is_bitwise_lossless_and_avoids_source_read(tmp_path, monkeypatch):
    import datasets.policy_image_cache as cache

    Image.fromarray(
        np.random.default_rng(1).integers(0, 256, (100, 160, 3), dtype=np.uint8)
    ).save(tmp_path / "image.png")
    original = policy_pixels(tmp_path, "image.png")
    cold = policy_pixels(tmp_path, "image.png", tmp_path / "cache")
    monkeypatch.setattr(
        cache,
        "img_path_to_data",
        lambda *args: (_ for _ in ()).throw(AssertionError("source opened")),
    )
    warm = policy_pixels(tmp_path, "image.png", tmp_path / "cache")
    torch.testing.assert_close(original, cold, rtol=0, atol=0)
    torch.testing.assert_close(original, warm, rtol=0, atol=0)


def test_legacy_sampler_reconstruction_matches_real_original_loaders():
    torch.manual_seed(285)
    state = torch.get_rng_state()
    expected = legacy_first_epoch_order(state, 101)
    # These are exactly the original runner's iterator operations.
    list(DataLoader(range(64), batch_size=8))
    actual = torch.cat(list(DataLoader(range(101), batch_size=8, shuffle=True)))
    assert torch.equal(expected, actual)
    sampler = ResumableOrder(101)
    sampler.order = expected
    sampler.start = 40
    resumed = torch.cat(list(DataLoader(range(101), batch_size=8, sampler=sampler)))
    assert torch.equal(resumed, actual[40:])

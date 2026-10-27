"""Exercise the VAE cache without importing ComfyUI's startup side effects."""

import ast
import unittest
from pathlib import Path

import torch


MODULE = Path(__file__).resolve().parents[1] / "integration/infinitetalk_infer.py"
tree = ast.parse(MODULE.read_text())
function = next(
    node for node in tree.body
    if isinstance(node, ast.FunctionDef) and node.name == "enable_request_vae_cache"
)


class FakeVAE:
    def __init__(self):
        self.calls = 0
        self.clear_calls = 0
        self.model = self

    def clear_cache(self):
        self.clear_calls += 1

    def encode(self, videos, device, **kwargs):
        self.calls += 1
        return videos.sum(dim=2, keepdim=True).clone()


class CacheTests(unittest.TestCase):
    def setUp(self):
        namespace = {
            "torch": torch,
            "FRAME_WINDOW_SIZE": 49,
            "get_value_at_index": lambda value, index: value[index],
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(MODULE), "exec"), namespace)
        self.enable = namespace["enable_request_vae_cache"]
        self.vae = FakeVAE()
        self.models = {"wanvideovaeloader_129": (self.vae,)}
        self.video = torch.ones(1, 3, 49, 2, 2)

    def test_identical_padded_input_hits_and_returns_separate_latent(self):
        self.enable(self.models)
        first = self.vae.encode(self.video, "cpu")
        second = self.vae.encode(self.video.clone(), "cpu")
        self.assertEqual(self.vae.calls, 1)
        self.assertEqual(self.vae.clear_calls, 1)
        self.assertTrue(torch.equal(first, second))
        self.assertNotEqual(first.data_ptr(), second.data_ptr())

    def test_changed_or_short_input_misses(self):
        self.enable(self.models)
        self.vae.encode(self.video, "cpu")
        self.vae.encode(self.video + 1, "cpu")
        self.vae.encode(self.video[:, :, :9], "cpu")
        self.assertEqual(self.vae.calls, 3)

    def test_new_request_resets_cache(self):
        self.enable(self.models)
        self.vae.encode(self.video, "cpu")
        self.enable(self.models)
        self.vae.encode(self.video, "cpu")
        self.assertEqual(self.vae.calls, 2)


if __name__ == "__main__":
    unittest.main()

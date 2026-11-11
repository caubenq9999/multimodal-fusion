import unittest
import numpy as np

from five_way_1k import (
    _fixed_candidate_scores,
    _fixed_query_caption_scores,
    _fixed_query_diffusion_scores,
    _gate_report,
    make_adafuse,
)


class FiveWayTests(unittest.TestCase):
    def test_fixed_score_shapes_and_finiteness(self):
        rng = np.random.default_rng(42)
        q = rng.normal(size=(5, 8)).astype(np.float32)
        image = rng.normal(size=(7, 8)).astype(np.float32)
        caption = rng.normal(size=(7, 8)).astype(np.float32)
        diffusion = rng.normal(size=(5, 8)).astype(np.float32)
        for scores in [
            _fixed_candidate_scores(q, image, caption, .8),
            _fixed_query_diffusion_scores(q, diffusion, image, .7),
            _fixed_query_caption_scores(q, caption, image, .5, query_chunk=2),
        ]:
            self.assertEqual(scores.shape, (5, 7))
            self.assertTrue(np.isfinite(scores).all())

    def test_adafuse_pairwise_shapes_and_gradients(self):
        import torch
        from five_way_1k import _query_caption_logits

        torch.manual_seed(42)
        model = make_adafuse(16, projection=12, expert_hidden=32, dropout=0)
        q = torch.randn(4, 16)
        caption = torch.randn(6, 16)
        image = torch.randn(6, 16)
        logits = _query_caption_logits(model, q, caption, image, candidate_chunk=2)
        self.assertEqual(tuple(logits.shape), (4, 6))
        logits.square().mean().backward()
        self.assertGreater(model.out.weight.grad.abs().sum(), 0)
        self.assertGreater(model.reliability[0].weight.grad.abs().sum(), 0)

    def test_gate_report(self):
        report = _gate_report(
            np.asarray([[.2], [.8]]),
            np.asarray([[.1, .2, .3, .4], [.4, .3, .2, .1]]),
            "query", "diffusion",
        )
        self.assertEqual(report["count"], 2)
        self.assertAlmostEqual(report["gate_mean"], .5)
        self.assertIn("query", report["gate_semantics"])


if __name__ == "__main__":
    unittest.main()

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from lib.data.sampler import sample_position
from lib.train import _masked_mean, evaluate_policy


class ReplayContractTests(unittest.TestCase):
    def test_final_input_uses_appended_final_record(self):
        source = SimpleNamespace(
            is_final=False,
            final_moves_left=3.0,
            simulation=SimpleNamespace(includes_final=True),
            final_position=None,
        )
        last_non_final = SimpleNamespace(is_final=False)
        final = SimpleNamespace(is_final=True)
        positions = [SimpleNamespace(is_final=False) for _ in range(5)]
        positions.extend(
            [source, last_non_final, SimpleNamespace(is_final=False), final]
        )
        group = SimpleNamespace(positions=positions)

        with patch("lib.data.sampler.random.randrange", return_value=5):
            _, sampled = sample_position(
                group, include_final=False, include_final_for_each=True
            )

        self.assertIs(sampled.final_position, final)

    def test_policy_loss_rejects_excess_target_mass(self):
        with self.assertRaisesRegex(AssertionError, "Invalid value mass"):
            evaluate_policy(
                torch.zeros((1, 2)),
                torch.tensor([[0, 1]]),
                torch.tensor([[0.75, 0.75]]),
                mask_invalid_moves=True,
            )

    def test_masked_mean_uses_only_terminal_entries(self):
        values = torch.tensor([1.0, 10.0, 5.0])
        terminal = torch.tensor([True, False, True])

        self.assertEqual(_masked_mean(values, terminal).item(), 3.0)


if __name__ == "__main__":
    unittest.main()

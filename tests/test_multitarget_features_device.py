import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from peva_sim.belief_contract import CONTEXT_SCHEMA
from peva_sim.collect_simulated_belief import TARGET_SCHEMA
from peva_sim.env import Config, MaritimeSAR
from peva_sim.latent_state import TargetStateCodec
from peva_sim.multitarget_features import MultiTargetFeatureAdapter
from peva_sim.reference_transport import ReferenceTransportJointBelief


class MultiTargetFeatureDeviceTests(unittest.TestCase):
    def checkpoint(self, directory, config):
        model = ReferenceTransportJointBelief(
            16 + config.grid_side ** 2, TargetStateCodec(config.n_targets))
        metadata = {
            "architecture": model.architecture,
            "context_schema": CONTEXT_SCHEMA,
            "input_target_schema": TARGET_SCHEMA,
            "context_dim": 16 + config.grid_side ** 2,
            "latent_dim": 128,
            "n_targets": config.n_targets,
            "source_metadata": {"config": vars(config)},
        }
        path = Path(directory) / "aux.pt"
        torch.save({"model": model.state_dict(), "metadata": metadata}, path)
        return path

    def test_cpu_is_repeatable_and_reports_device(self):
        config = Config(n_targets=3, particles=10, grid_side=4,
                        packet_drop=1., coverage_mode="intensity")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.checkpoint(directory, config)
            outputs = []
            for _ in range(2):
                env = MaritimeSAR(config)
                observation, _ = env.reset(8)
                adapter = MultiTargetFeatureAdapter(
                    checkpoint, config, seed=3, device="cpu")
                outputs.append(adapter.step(env, observation))
                self.assertEqual(next(adapter.model.parameters()).device.type, "cpu")
            self.assertTrue(np.array_equal(outputs[0], outputs[1]))
            self.assertEqual(
                outputs[0].shape,
                (config.n_uavs,
                 176 + 3 * config.n_targets + config.n_uavs),
            )
            self.assertTrue(np.array_equal(outputs[0][:, -config.n_uavs:],
                                           np.eye(config.n_uavs)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_executes_entire_neural_path(self):
        config = Config(n_targets=3, particles=10, grid_side=4,
                        packet_drop=1., coverage_mode="intensity")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.checkpoint(directory, config)
            env = MaritimeSAR(config)
            observation, _ = env.reset(8)
            adapter = MultiTargetFeatureAdapter(
                checkpoint, config, seed=3, device="cuda")
            output = adapter.step(env, observation)
            self.assertEqual(next(adapter.model.parameters()).device.type, "cuda")
            self.assertEqual(
                output.shape,
                (config.n_uavs,
                 176 + 3 * config.n_targets + config.n_uavs),
            )
            self.assertTrue(np.isfinite(output).all())
            self.assertTrue(np.isfinite(adapter.last["belief_samples"]).all())


    def test_deterministic_and_physics_context_ablations_are_effective(self):
        config = Config(n_targets=3, particles=10, grid_side=4,
                        packet_drop=1., coverage_mode="intensity")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.checkpoint(directory, config)
            outputs = []
            for seed in (3, 900):
                env = MaritimeSAR(config)
                observation, _ = env.reset(8)
                adapter = MultiTargetFeatureAdapter(
                    checkpoint, config, seed=seed, samples=1,
                    deterministic=True, physics_context=False, device="cpu")
                outputs.append(adapter.step(env, observation))
                self.assertTrue((adapter.last["spread"] == 0).all())
                self.assertFalse(adapter.last["physics_context"])
                self.assertTrue((adapter.last["context"][:, 15:] == 0).all())
            self.assertTrue(np.array_equal(outputs[0], outputs[1]))

if __name__ == "__main__":
    unittest.main()

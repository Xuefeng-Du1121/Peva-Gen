import pytest
import torch

from peva_sim.formal_mappo import RecurrentActor


def _zero_module(module):
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.zero_()


def test_shared_head_role_bias_selects_distinct_action_means():
    actor = RecurrentActor(obs_dim=8, hidden_dim=4, role_dim=3)
    _zero_module(actor)
    with torch.no_grad():
        actor.role_bias.copy_(torch.tensor([
            [1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]))
    obs = torch.zeros(3, 8)
    obs[:, -3:] = torch.eye(3)
    mean, _, _ = actor.step(
        obs, actor.initial_state(3), torch.zeros(3, 1))
    assert torch.equal(mean, torch.tensor([
        [1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]))


def test_role_conditioned_actor_rejects_missing_role():
    actor = RecurrentActor(obs_dim=8, hidden_dim=4, role_dim=3)
    with pytest.raises(ValueError, match="one-hot"):
        actor.step(torch.zeros(2, 8), actor.initial_state(2),
                   torch.zeros(2, 1))


def test_shared_head_remains_available_without_roles():
    actor = RecurrentActor(obs_dim=5, hidden_dim=4)
    obs = torch.zeros(2, 5)
    mean, _, hidden = actor.step(
        obs, actor.initial_state(2), torch.zeros(2, 1))
    assert mean.shape == (2, 2)
    assert hidden.shape == (2, 4)

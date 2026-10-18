"""Numerical and integration checks; this is not a convergence benchmark."""
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from cnn_networks import DuelingCNNNetwork
from dqn import DQNAgent, ReplayMemory, PrioritizedReplayMemory, Transition
from networks import DuelingMLPNetwork
import utils

torch.set_num_threads(1)


def tiny_agent(double=False, terminal=False):
    agent = DQNAgent.__new__(DQNAgent)
    agent.batch_size = 1
    agent.device = torch.device("cpu")
    agent.is_cnn = False
    agent.env = SimpleNamespace(observation_space=gym.spaces.Discrete(2))
    agent.gamma = 0.5
    agent.use_double_dqn = double
    agent.use_per = False
    agent.policy_net = nn.Linear(2, 2, bias=False)
    agent.target_net = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        # At next_state=1 the online and target networks disagree on the best action.
        agent.policy_net.weight.copy_(torch.tensor([[0., 9.], [0., 1.]]))
        agent.target_net.weight.copy_(torch.tensor([[0., 2.], [0., 7.]]))
    agent.optimizer = torch.optim.SGD(agent.policy_net.parameters(), lr=0.0)
    agent.memory = ReplayMemory(4)
    agent.memory.push(0, 0, None if terminal else 1, torch.tensor([1.]))
    return agent


class SolutionChecks(unittest.TestCase):
    def test_uniform_replay_capacity_and_sample(self):
        memory = ReplayMemory(3)
        for i in range(5):
            memory.push(i, 0, None, torch.tensor([0.]))
        self.assertEqual([t.state for t in memory.memory], [2, 3, 4])
        self.assertEqual({t.state for t in memory.sample(3)}, {2, 3, 4})

    def test_per_sampling_probabilities_and_weights(self):
        memory = PrioritizedReplayMemory(3, alpha=1.0, beta=1.0)
        for i in range(3):
            memory.push(i, 0, None, torch.tensor([0.]))
        memory.update_priorities([0, 1, 2], np.array([[1.], [-2.], [4.]]))
        np.random.seed(12)
        _, indices, weights = memory.sample(12000)
        expected = memory.priorities / memory.priorities.sum()
        observed = np.bincount(indices, minlength=3) / len(indices)
        np.testing.assert_allclose(observed, expected, atol=0.015)
        # For beta=1, weight * sampling_probability is constant within the batch.
        corrected = weights.numpy() * expected[indices]
        np.testing.assert_allclose(corrected, np.repeat(corrected[0], len(indices)), rtol=1e-5)
        self.assertAlmostEqual(weights.max().item(), 1.0)
        memory.alpha = 0.0
        memory.beta = 0.0
        _, indices, weights = memory.sample(12000)
        np.testing.assert_allclose(np.bincount(indices, minlength=3) / len(indices), [1/3]*3, atol=0.015)
        self.assertTrue(torch.equal(weights, torch.ones_like(weights)))
        memory.push(3, 0, None, torch.tensor([0.]))
        self.assertEqual(memory.memory[0].state, 3)
        self.assertGreater(memory.priorities[0], 0)

    def test_double_target_separates_selection_and_evaluation(self):
        # DQN: y=1+0.5*7=4.5 -> Huber(0,4.5)=4.0.
        # DDQN: online chooses action 0; y=1+0.5*2=2 -> Huber=1.5.
        for double, expected in [(False, 4.0), (True, 1.5)]:
            agent = tiny_agent(double=double)
            self.assertAlmostEqual(agent.optimize_model(), expected)
            self.assertIsNone(agent.target_net.weight.grad)
            self.assertEqual(agent.policy_net.weight.grad[0, 1].item(), 0.0)

    def test_all_terminal_batch_has_no_bootstrap(self):
        for double in (False, True):
            self.assertAlmostEqual(tiny_agent(double, terminal=True).optimize_model(), 0.5)

    def test_per_weighted_loss_is_per_sample(self):
        agent = tiny_agent()
        agent.use_per = True
        agent.batch_size = 2
        samples = [Transition(0, 0, None, torch.tensor([r])) for r in [0.5, 3.0]]
        class FixedMemory:
            def __len__(self): return 2
            def sample(self, size): return samples, np.array([0, 1]), torch.tensor([1., 0.1])
            def update_priorities(self, indices, errors): self.errors = errors.reshape(-1)
        agent.memory = FixedMemory()
        # (1 * 0.125 + 0.1 * 2.5) / 2 = 0.1875.
        self.assertAlmostEqual(agent.optimize_model(), 0.1875, places=6)
        np.testing.assert_allclose(agent.memory.errors, [0.5, 3.0])

    def test_dueling_mean_and_advantage_shift_invariance(self):
        torch.manual_seed(2)
        for model, x, head in [
            (DuelingMLPNetwork(4, 3, hidden_size=16), torch.randn(2, 4), "advantage_head"),
            (DuelingCNNNetwork((12, 84, 84), 5, hidden_size=16), torch.randn(2, 12, 84, 84).to(memory_format=torch.channels_last), "advantage_stream"),
        ]:
            q = model(x)
            if head == "advantage_head":
                value = model.value_head(model.feature_layer(x))
            else:
                features = model.features(x).flatten(1)
                value = model.value_stream(features)
            torch.testing.assert_close(q.mean(1, keepdim=True), value)
            with torch.no_grad():
                getattr(model, head)[-1].bias.add_(5.0)
            torch.testing.assert_close(model(x), q, atol=1e-6, rtol=1e-5)

    def test_training_preserves_truncation_next_state(self):
        class OneStep(gym.Env):
            observation_space = gym.spaces.Box(-2, 2, shape=(4,), dtype=np.float32)
            action_space = gym.spaces.Discrete(2)
            def __init__(self, terminal): self.terminal = terminal
            def reset(self, *, seed=None, options=None): return np.zeros(4, dtype=np.float32), {}
            def step(self, action): return np.ones(4, dtype=np.float32), 1., self.terminal, not self.terminal, {}
        for terminal in (False, True):
            with tempfile.TemporaryDirectory() as output:
                agent = DQNAgent(episodes=1, batch_size=8, replay_size=8,
                                 output_root=output, record_video=False)
                agent.env.close()
                agent.env = OneStep(terminal)
                with patch("dqn.utils.evaluate_policy", return_value=0.0):
                    agent.train()
                self.assertEqual(agent.memory.memory[0].next_state is None, terminal)

    def test_cnn_evaluation_uses_normalized_stacked_input(self):
        class Pixels(gym.Env):
            observation_space = gym.spaces.Box(0, 255, shape=(96, 96, 3), dtype=np.uint8)
            action_space = gym.spaces.Discrete(5)
            def reset(self, *, seed=None, options=None): return np.full((96, 96, 3), 255, dtype=np.uint8), {}
            def step(self, action): return self.reset()[0], 1., True, False, {}
        class Capture(nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = nn.Parameter(torch.zeros(5))
            def forward(self, x):
                self.seen = x.detach()
                return self.bias.unsqueeze(0)
        policy = Capture()
        with patch("utils.gym.make", return_value=Pixels()):
            reward = utils.evaluate_policy(policy, None, "CarRacing-v2", episodes=1, seed=0)
        self.assertEqual(reward, 1.0)
        self.assertEqual(tuple(policy.seen.shape), (1, 12, 84, 84))
        self.assertEqual(policy.seen.max().item(), 1.0)
        self.assertTrue(policy.training)


if __name__ == "__main__":
    unittest.main(verbosity=2)

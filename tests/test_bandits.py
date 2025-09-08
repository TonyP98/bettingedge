import numpy as np

from engine.online.bandits import LinUCB, ThompsonGaussian
from engine.eval.replay import crra_reward


def test_linucb_converges():
    np.random.seed(0)
    actions = [(0.01, 0.1), (0.02, 0.2)]
    bandit = LinUCB(actions, alpha=1.0, epsilon=0.0, prior_lambda=1.0)
    correct = 0
    n = 1000
    for _ in range(n):
        x = np.random.randn(2)
        best_idx = 0 if x[0] > x[1] else 1
        sel = bandit.select(x)["action"]
        reward = x[best_idx] + np.random.randn() * 0.1
        bandit.update(x, sel, reward)
        if sel == actions[best_idx]:
            correct += 1
    assert correct / n > 0.6


def test_thompson_sampling_updates():
    np.random.seed(1)
    actions = [(0.01, 0.1), (0.02, 0.2)]
    bandit = ThompsonGaussian(actions, epsilon=0.5, prior_lambda=1.0, seed=1)
    counts = {a: 0 for a in actions}
    for _ in range(100):
        x = np.random.randn(2)
        sel = bandit.select(x)["action"]
        reward = np.random.randn()
        bandit.update(x, sel, reward)
        counts[sel] += 1
    assert all(c > 0 for c in counts.values())
    for a in actions:
        A = bandit.A[a]
        b = bandit.b[a]
        theta = np.linalg.solve(A, b)
        assert not np.isnan(theta).any()


def test_crra_reward_monotonic():
    r_pos = crra_reward(0.1, 1.0, 2.0)
    r_neg = crra_reward(-0.1, 1.0, 2.0)
    assert r_pos > r_neg
    assert -1.0 <= r_pos <= 1.0
    assert -1.0 <= r_neg <= 1.0

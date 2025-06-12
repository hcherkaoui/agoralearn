"""Simple time serie illustration."""

import numpy as np
import matplotlib.pyplot as plt


###############################################################################
# Globals
def state_features(t, d):
    """Return the states (feature vectors)."""
    features = [t**i for i in range(min(d, 4))]
    features += [np.sin(k * t * np.pi) for k in range(1, 3) if len(features) < d]
    features += [np.cos(k * t * np.pi) for k in range(1, 3) if len(features) < d]
    features += [1.0] if len(features) < d else []

    while len(features) < d:
        features.append(np.random.randn() * 0.01)

    return np.array(features[:d])


def generate_markov_chain(n, P, states):
    """Generate the state serie."""
    X = []
    idx = np.random.choice(len(states))
    for _ in range(n):
        X.append(states[idx])
        idx = np.random.choice(len(states), p=P[idx])
    return np.array(X)


###############################################################################
# Settings
figsize = (8, 4)
fontsize = 14
lw = 3.0

d = 6
assert d >= 6, f"Dimension d should be higher than 6, got {d}"
n_states = 10
n_samples = 300
sigma = 1.0
alpha = 2.0
np.random.seed(None)

theta = np.zeros(d)
theta[:min(d, 6)] = [0.03, 0.01, -0.02, 0.015, 0.01, -0.01]

states = [state_features(t, d) for t in alpha * np.linspace(-1, 1, n_states)]

P = np.zeros((n_states, n_states))
for i in range(n_states):
    P[i, i] = 0.6
    if i > 0:
        P[i, i - 1] += 0.2
    if i < n_states - 1:
        P[i, i + 1] += 0.2
    P[i] /= P[i].sum()

###############################################################################
# Main
print("[Main] Simple time serie example")
X = generate_markov_chain(n_samples, P, states)
prices = np.cumsum(X @ theta + np.random.randn(n_samples) * sigma)

###############################################################################
# Plotting
print("[Main] Plotting")
plt.figure(figsize=(6, 3))

plt.plot(prices, lw=lw, alpha=0.8)

plt.xlabel("Time", fontsize=fontsize)
plt.ylabel("Relative stock value", fontsize=fontsize)
plt.legend(fontsize=int(0.5*fontsize))
plt.grid(True)
plt.tight_layout()

plt.show()

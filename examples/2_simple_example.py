"""Simple usage example."""

# Authors: Hamza Cherkaoui

import numpy as np
from agoralearn.forecasters import OracleForecaster
from agoralearn.transition_matrix import generate_transition_matrix
from agoralearn.utils import simulate_hmm


###############################################################################
# Settings
d = 10
n_states = 50
states = list(np.random.randn(n_states, d))
P = generate_transition_matrix(n_states, type="cyclic")

theta_star = np.array([1] + [0] * (d-1))
X_train, y_train = simulate_hmm(n=100, d=d, theta_star=theta_star, P=P, states=states, sigma=0.1)
X_test, y_test = simulate_hmm(n=100, d=d, theta_star=theta_star, P=P, states=states, sigma=0.1)

###############################################################################
# Main
print("[Main] Simple forecasting example")
model = OracleForecaster(states=states, P=P).batch_fit(X_train, y_train)
predictions = [model.online_predict(x) for x in X_test]
print(f"[Main] Err. test: {np.linalg.norm(predictions - y_test):.2f}")
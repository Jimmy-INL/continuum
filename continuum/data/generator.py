import numpy as np
import pandas as pd
from sklearn.datasets import make_regression


def make_sarsa_frame(
    n_samples: int = 10000,
    n_features: int = 40,
    num_actions: int = 2,
    state_name: str = "state",
    action_name: str = "actions",
    response_name: str = "reward"
):
    state_x, reward_y = make_regression(
        n_samples=n_samples, n_features=n_features
    )
    xl = len(state_x[0, :])
    state_x, action_z = np.hsplit(state_x, [xl - num_actions])

    x_len = len(state_x)
    actions = [action_z[i] for i in range(x_len)]
    state = [state_x[i] for i in range(x_len)]
    reward = [reward_y[i] for i in range(x_len)]
    state_act: pd.DataFrame = pd.DataFrame({
        f"{state_name}": state,
        f"{action_name}": actions,
        f"{response_name}": reward
    })
    return state_act
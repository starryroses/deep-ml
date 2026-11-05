
import numpy as np

def rnn_forward(input_sequence: list[list[float]], initial_hidden_state: list[float], Wx: list[list[float]], Wh: list[list[float]], b: list[float]) -> list[float]:

    hidden_state = np.array(initial_hidden_state)
    Wx = np.array(Wx)
    Wh = np.array(Wh)
    b = np.array(b)

    for x in input_sequence:
        x = np.array(x)

        hidden_state = np.tanh(
            np.dot(Wx, x) +
            np.dot(Wh, hidden_state) +
            b
        )

    final_hidden_state = np.round(hidden_state, 4).tolist()

    return final_hidden_state

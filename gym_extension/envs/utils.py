from typing import Tuple

import numpy as np


def coordinate_to_position(coord: np.ndarray, playground_shape: np.ndarray) -> np.ndarray:
    """
    :param coord:
    :param playground_shape: like (n, m)
    :return:
    """
    x = coord[..., 0]
    y = coord[..., 1]
    # (..., n, 1)
    x_one_hot = np.eye(playground_shape[0], dtype=int)[x][..., np.newaxis]
    # (..., 1, m)
    y_one_hot = np.eye(playground_shape[1], dtype=int)[y][..., np.newaxis, :]
    # (..., n, m)
    return x_one_hot * y_one_hot


def position_to_coordinate(position: np.ndarray) -> np.ndarray:
    x = position.any(axis=-1).argmax(axis=-1)
    y = position.any(axis=-2).argmax(axis=-1)
    return np.stack((x, y), axis=-1)


if __name__ == "__main__":
    p = coordinate_to_position(
        np.array([[1, 4],
                  [3, 1]]),
        np.array([5, 5])
    )

    c = position_to_coordinate(p)

    print(c, p)

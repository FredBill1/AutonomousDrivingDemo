from typing import Any

import numpy as np
import numpy.typing as npt

from ...modeling.Car import Car
from ...modeling.Obstacles import Obstacles


def generate_car(obstacles: Obstacles) -> npt.NDArray[np.floating[Any]]:
    coords = obstacles.coordinates
    minx, maxx, miny, maxy = coords[:, 0].min(), coords[:, 0].max(), coords[:, 1].min(), coords[:, 1].max()
    state = np.random.uniform((minx, miny, -np.pi), (maxx, maxy, np.pi))
    while Car(*state).check_collision(obstacles):
        state = np.random.uniform((minx, miny, -np.pi), (maxx, maxy, np.pi))
    return state

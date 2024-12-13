from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.interpolate

from ..constants import *
from ..local_planner.ModelPredictiveControl import ModelPredictiveControl
from ..modeling.Car import Car
from .utils.plot_car import plot_car

TRAJECTORY_LENGTH = 100.0  # [m]

P_SWITCH_DIRECTION = 0.02
P_CHANGE_STEER = 0.2


def _generate_random_trajectory() -> npt.NDArray[np.floating[Any]]:
    car = Car(0, 0, np.random.uniform(-np.pi, np.pi))
    car.velocity = 1.0 if np.random.rand() > 0.5 else -1.0
    car.steer = np.random.uniform(-Car.MAX_STEER, Car.MAX_STEER)
    trajectory = []
    for _ in range(round(TRAJECTORY_LENGTH / MOTION_RESOLUTION)):
        if len(trajectory) > 1 and np.random.rand() < P_SWITCH_DIRECTION:
            car.velocity *= -1
        if np.random.rand() < P_CHANGE_STEER:
            car.steer = np.random.uniform(-Car.MAX_STEER, Car.MAX_STEER)
        car.update(MOTION_RESOLUTION)
        trajectory.append([car.x, car.y, car.yaw, car.velocity])
    return np.array(trajectory)


def main(ax: plt.Axes) -> None:
    trajectory = _generate_random_trajectory()

    ax.cla()
    ax.grid()
    ax.plot(*trajectory.T[:2], "-r")
    ax.set_aspect("equal", "datalim")
    ax.relim()
    ax.autoscale_view()
    ax.autoscale(False)

    timestamp_s = 0.0
    car = Car(*trajectory[0, :3])
    mpc = ModelPredictiveControl(trajectory)
    car_artists = plot_car(car, ax)
    states_artist = ax.plot([], [], "-b")[0]
    ref_states_artist = ax.plot([], [], "xg")[0]
    while True:
        plt.draw()
        if plt.waitforbuttonpress(0.05) is not None:
            break

        res = mpc.update(car, LOCAL_PLANNER_DELTA_TIME)

        timestamps = np.arange(len(res.controls)) * LOCAL_PLANNER_DELTA_TIME + timestamp_s
        velocities = car.velocity + np.cumsum(res.controls[:, 0] * LOCAL_PLANNER_DELTA_TIME)
        steers = res.controls[:, 1]
        controls = np.column_stack((velocities, steers))
        tck, _ = scipy.interpolate.splprep(controls.T, s=0, k=1, u=timestamps)
        for _ in range(round(LOCAL_PLANNER_DELTA_TIME / SIMULATION_DELTA_TIME)):
            timestamp_s += SIMULATION_DELTA_TIME
            velocity, steer = scipy.interpolate.splev(timestamp_s, tck)
            car.update_with_control(velocity, steer, SIMULATION_DELTA_TIME)

        ax.title.set_text(
            f"Timestamp: {timestamp_s:.1f}s "
            f"Velocity: {car.velocity * 3.6:.1f}km/h "
            f"Steer: {np.rad2deg(car.steer):.1f}° "
            "click to reset"
        )

        for artist in car_artists:
            artist.remove()
        car_artists = plot_car(car, ax)
        states_artist.set_data(*res.states.T[:2])
        ref_states_artist.set_data(*res.ref_states.T[:2])


if __name__ == "__main__":
    fig, ax = plt.subplots()
    fig.canvas.mpl_connect("close_event", lambda _: quit())
    while True:
        main(ax)

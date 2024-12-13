import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
from matplotlib.collections import LineCollection

from ..constants import *
from ..global_planner.hybrid_a_star import Node, hybrid_a_star
from ..local_planner.ModelPredictiveControl import ModelPredictiveControl
from ..modeling.Car import Car
from ..modeling.Obstacles import Obstacles
from .utils.generate_car import generate_car
from .utils.generate_obstacle_coordnates import generate_obstacle_coordnates
from .utils.plot_car import plot_car


def main(ax: plt.Axes) -> None:
    obstacle_coordinates = generate_obstacle_coordnates()
    obstacles = Obstacles(obstacle_coordinates)
    start = generate_car(obstacles)
    goal = generate_car(obstacles)

    ax.cla()
    ax.grid()
    ax.plot(*obstacle_coordinates.T, ".r")
    plot_car(Car(*start), ax, color="b")
    plot_car(Car(*goal), ax, color="g")
    ax.set_aspect("equal", "datalim")
    ax.title.set_text("Hybrid A* Planning")
    ax.relim()
    ax.autoscale_view()
    ax.autoscale(False)

    explored_nodes_artists: list[LineCollection] = []
    segments: list[Node] = []

    def update_segments(node: Node) -> bool:
        segments.append(node.get_plot_trajectory()[:, :2])
        if len(segments) > GLOBAL_PLANNER_SEGMENT_COLLECTION_SIZE:
            explored_nodes_artists.append(ax.add_collection(LineCollection(segments, colors="b", linewidths=0.5)))
            segments.clear()
            plt.pause(0.01)
        return False

    trajectory = hybrid_a_star(start, goal, obstacles, cancel_callback=update_segments)

    for artist in explored_nodes_artists:
        artist.remove()

    if trajectory is None:
        ax.title.set_text(f"Hybrid A* Planning: 'Failure', click to reset")
        plt.draw()
        plt.waitforbuttonpress()
        return

    ax.plot(*trajectory.T[:2], "-r")

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

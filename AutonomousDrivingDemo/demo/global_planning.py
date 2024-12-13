import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from ..constants import *
from ..global_planner.hybrid_a_star import Node, hybrid_a_star
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
        if len(segments) >= GLOBAL_PLANNER_SEGMENT_COLLECTION_SIZE:
            explored_nodes_artists.append(ax.add_collection(LineCollection(segments, colors="b", linewidths=0.5)))
            segments.clear()
            plt.pause(0.01)
        return False

    trajectory = hybrid_a_star(start, goal, obstacles, cancel_callback=update_segments)

    for artist in explored_nodes_artists:
        artist.remove()

    if trajectory is not None:
        ax.plot(*trajectory.T[:2], "-r")

    ax.title.set_text(f"Hybrid A* Planning: {'Success' if trajectory is not None else 'Failure'}, click to reset")
    plt.draw()
    plt.waitforbuttonpress()


if __name__ == "__main__":
    fig, ax = plt.subplots()
    fig.canvas.mpl_connect("close_event", lambda _: quit())
    while True:
        main(ax)

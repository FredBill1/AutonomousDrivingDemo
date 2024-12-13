from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.backend_bases import MouseEvent

from ..MapServerNode import MapServerNode
from ..modeling.Car import Car
from ..modeling.Obstacles import Obstacles
from .utils.plot_car import plot_car


def main(fig: plt.Figure, ax: plt.Axes) -> None:
    map_server_node = MapServerNode()
    map_server_node.init()
    car = Car(0, 0, 0)
    map_server_node.update(0.0, car)
    known_obstacles = Obstacles(map_server_node.known_obstacle_coordinates)

    ax.cla()
    ax.grid()
    ax.set_aspect("equal", "datalim")
    ax.plot(*map_server_node.unknown_obstacle_coordinates.T, ".b")[0]
    known_obstacles_artist = ax.plot(*map_server_node.known_obstacle_coordinates.T, ".r")[0]
    ax.relim()
    ax.autoscale_view()
    ax.autoscale(False)
    ax.title.set_text("Move the car with the mouse, click to reset")
    car_artists = plot_car(car, ax, color="c" if car.check_collision(known_obstacles) else "k", with_lidar=True)

    history_poses = deque([(car.x, car.y)], maxlen=10)

    def mouse_move(event: MouseEvent) -> None:
        nonlocal car_artists
        if event.xdata is not None:
            history_poses.append((event.xdata, event.ydata))
            car.yaw = np.arctan2(event.ydata - history_poses[0][1], event.xdata - history_poses[0][0])
            car.x, car.y = event.xdata, event.ydata
            map_server_node.update(0.0, car)

            for artist in car_artists:
                artist.remove()
            car_artists = plot_car(car, ax, color="c" if car.check_collision(known_obstacles) else "k", with_lidar=True)
            plt.draw()

    def known_obstacle_coordinates_updated(known_obstacle_coordinates: npt.NDArray[np.floating]) -> None:
        nonlocal known_obstacles
        known_obstacles_artist.set_data(*known_obstacle_coordinates.T)
        known_obstacles = Obstacles(known_obstacle_coordinates)
        plt.draw()

    map_server_node.known_obstacle_coordinates_updated.connect(known_obstacle_coordinates_updated)
    cid = fig.canvas.mpl_connect("motion_notify_event", mouse_move)
    plt.draw()
    plt.waitforbuttonpress()
    fig.canvas.mpl_disconnect(cid)


if __name__ == "__main__":
    fig, ax = plt.subplots()
    fig.canvas.mpl_connect("close_event", lambda _: quit())
    while True:
        main(fig, ax)

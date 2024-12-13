import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.backend_bases import KeyEvent
from matplotlib.widgets import Slider

from ..constants import *
from ..modeling.Car import Car
from ..modeling.Obstacles import Obstacles
from .utils.generate_obstacle_coordnates import generate_obstacle_coordnates
from .utils.plot_car import plot_car

ANIMATION_INTERVAL = 0.1  # [s]


def main() -> None:
    car = Car(0, 0, 0)
    obstacle_coordinates = generate_obstacle_coordnates()
    obstacles = Obstacles(obstacle_coordinates)

    fig, ax = plt.subplots()
    fig.canvas.mpl_connect("close_event", lambda _: quit())
    ax.cla()
    ax.grid()
    ax.plot(*obstacle_coordinates.T, ".r")
    ax.set_aspect("equal", "datalim")
    ax.relim()
    ax.autoscale_view()
    ax.autoscale(False)
    ax.title.set_text("Use 'w' and 's' to control the car")
    car_artists = plot_car(car, ax, color="c" if car.check_collision(obstacles) else "k")

    pressed_keys = set()

    def on_press(event: KeyEvent) -> None:
        pressed_keys.add(event.key)

    def on_release(event: KeyEvent) -> None:
        pressed_keys.discard(event.key)

    def func(*_) -> None:
        target_velocity = 0
        if "w" in pressed_keys:
            target_velocity = Car.MAX_SPEED
        elif "s" in pressed_keys:
            target_velocity = -Car.MAX_SPEED
        for _ in range(round(ANIMATION_INTERVAL / SIMULATION_DELTA_TIME)):
            car.update_with_control(target_velocity, -np.deg2rad(steer_slider.val), SIMULATION_DELTA_TIME)

        nonlocal car_artists
        for artist in car_artists:
            artist.remove()
        car_artists = plot_car(car, ax, color="c" if car.check_collision(obstacles) else "k")

    plt.rcParams["keymap.save"].remove("s")
    ax_steer_slider = fig.add_axes([0.1, 0.01, 0.8, 0.03])
    steer_slider = Slider(ax_steer_slider, "Steer", -np.rad2deg(Car.MAX_STEER), np.rad2deg(Car.MAX_STEER), valinit=0)
    fig.canvas.mpl_connect("key_press_event", on_press)
    fig.canvas.mpl_connect("key_release_event", on_release)

    anim = FuncAnimation(fig, func, save_count=0, interval=round(ANIMATION_INTERVAL * 1000))
    anim  # keep reference
    plt.show()


if __name__ == "__main__":
    main()

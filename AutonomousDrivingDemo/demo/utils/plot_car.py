import matplotlib.pyplot as plt
import numpy as np

from ...modeling.Car import Car


def plot_car(car: Car, ax: plt.Axes, *, color="k", with_lidar: bool = False) -> list[plt.Artist]:
    BOX = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1], [-1, -1]]) / 2
    outline = BOX * [Car.LENGTH, Car.WIDTH] + [Car.LENGTH / 2 - Car.BACK_TO_WHEEL, 0]
    wheel = BOX * [Car.WHEEL_LENGTH, Car.WHEEL_WIDTH]

    cy, sy = np.cos(car.yaw), np.sin(car.yaw)
    cs, ss = np.cos(car.steer), np.sin(car.steer)
    rot1 = np.array([[cy, -sy], [sy, cy]])
    rot2 = np.array([[cs, -ss], [ss, cs]])
    f_wheel = (rot2 @ wheel.T).T
    fl_wheel = f_wheel + [Car.WHEEL_BASE, Car.WHEEL_SPACING / 2]
    fr_wheel = f_wheel + [Car.WHEEL_BASE, -Car.WHEEL_SPACING / 2]
    rl_wheel = wheel + [0, Car.WHEEL_SPACING / 2]
    rr_wheel = wheel + [0, -Car.WHEEL_SPACING / 2]

    artists = []
    for box in (outline, fl_wheel, fr_wheel, rl_wheel, rr_wheel):
        box = (rot1 @ box.T).T + [car.x, car.y]
        artists.extend(ax.plot(*box.T, "-", color=color))
    artists.extend(ax.plot(car.x, car.y, "*", color=color))
    if with_lidar:
        x, y = car.x + cy * Car.BACK_TO_CENTER, car.y + sy * Car.BACK_TO_CENTER
        artists.append(ax.add_artist(plt.Circle((x, y), Car.SCAN_RADIUS, color=color, fill=False)))
    return artists

import PySide6

"set PySide6 backend"

from collections import deque
from typing import Any, Optional, override

import numpy as np
import numpy.typing as npt
import pyqtgraph as pg
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.GraphicsScene.mouseEvents import MouseDragEvent
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QMainWindow

from .CarSimulationNode import CarSimulationNode
from .constants import *
from .GlobalPlannerNode import GlobalPlannerNode
from .LocalPlannerNode import LocalPlannerNode, LocalPlanningTrajectories
from .MapServerNode import MapServerNode
from .modeling.Car import Car
from .modeling.Obstacles import Obstacles
from .plotting.CarItem import CarItem
from .TrajectoryCollisionCheckingNode import TrajectoryCollisionCheckingNode
from .ui.mainwindow_ui import Ui_MainWindow

SIMULATION_INTERVAL = 0.02  # [s]
SIMULATION_PUBLISH_INTERVAL = 0.05  # [s]

LOCAL_PLANNER_UPDATE_INTERVAL = 0.1  # [s]

DASHBOARD_HISTORY_SIZE = 500

REPLAN_MAX_SPEED = 5 / 3.6  # [m/s]


class _CustomViewBox(pg.ViewBox):
    sigMouseDrag = Signal(MouseDragEvent)

    @override
    def mouseDragEvent(self, ev: MouseDragEvent) -> None:
        if ev.button() != Qt.MouseButton.LeftButton:
            return super().mouseDragEvent(ev)
        ev.accept()
        self.sigMouseDrag.emit(ev)


class MainWindow(QMainWindow):
    set_state = Signal(Car)
    canceled = Signal()
    set_goal = Signal(Car, Car, Obstacles)
    braked = Signal()
    restarted = Signal()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # prepare data
        self._measured_state: Optional[Car] = None
        self._goal_state: Optional[Car] = None
        self._measured_timestamp = 0.0
        self._measured_velocities: deque[float] = deque([0.0], maxlen=DASHBOARD_HISTORY_SIZE)
        self._measured_steers: deque[float] = deque([0.0], maxlen=DASHBOARD_HISTORY_SIZE)
        self._measured_timestamps: deque[float] = deque([0.0], maxlen=DASHBOARD_HISTORY_SIZE)
        self._brake_trajectory: Optional[npt.NDArray[np.floating[Any]]] = None
        self._local_planning = False
        self._minx = self._maxx = self._miny = self._maxy = 0.0

        # setup ui
        self._ui = Ui_MainWindow()
        self._ui.setupUi(self)

        self._plot_viewbox = _CustomViewBox(enableMenu=False)
        self._plot_viewbox.setAspectLocked()
        self._plot_widget = pg.PlotWidget(viewBox=self._plot_viewbox, title="Timestamp: 0.0s")
        self._plot_widget.addItem(pg.GridItem())

        self._velocity_plot_widget = pg.PlotWidget(title="Velocity: 0.0km/h")
        self._velocity_plot_widget.disableAutoRange(axis=pg.ViewBox.YAxis)
        self._velocity_plot_widget.setYRange(Car.MIN_SPEED * 3.6, Car.MAX_SPEED * 3.6)
        self._velocity_plot_widget.addItem(pg.GridItem())
        self._steer_plot_widget = pg.PlotWidget(title="Steer: 0.0°")
        self._steer_plot_widget.disableAutoRange(axis=pg.ViewBox.YAxis)
        self._steer_plot_widget.setYRange(-np.rad2deg(Car.MAX_STEER), np.rad2deg(Car.MAX_STEER))
        self._steer_plot_widget.addItem(pg.GridItem())

        # docks
        self._visualization_dock = Dock("Visualization", size=(4, 2))
        self._velocity_plot_dock = Dock("Velocity", size=(2, 2))
        self._steer_plot_dock = Dock("Steer", size=(2, 2))
        self._visualization_dock.addWidget(self._plot_widget)
        self._velocity_plot_dock.addWidget(self._velocity_plot_widget)
        self._steer_plot_dock.addWidget(self._steer_plot_widget)
        self._ui.dockarea.addDock(self._visualization_dock, "left")
        self._ui.dockarea.addDock(self._velocity_plot_dock, "right", self._visualization_dock)
        self._ui.dockarea.addDock(self._steer_plot_dock, "bottom", self._velocity_plot_dock)

        # graphics items
        self._bounding_box_item = pg.PlotCurveItem(pen=pg.mkPen("r"))
        self._known_obstacles_item = pg.ScatterPlotItem(size=3, symbol="o", pen=None, brush=(255, 0, 0))
        self._unknown_obstacles_item = pg.ScatterPlotItem(size=3, symbol="o", pen=None, brush=(0, 255, 255))
        self._measured_state_item = CarItem(None, color="w", with_lidar=True)
        self._pressed_pose_item = CarItem(None, color="g")
        self._pressed_pose_item.setVisible(False)
        self._goal_pose_item = CarItem(None, color="g")
        self._goal_pose_item.setVisible(False)
        self._goal_unreachable_item = pg.TextItem("Goal is unreachable", color="r")
        font = QFont()
        font.setPointSize(15)
        self._goal_unreachable_item.setFont(font)
        self._goal_unreachable_item.setVisible(False)
        self._local_trajectory_item = pg.PlotCurveItem(pen=pg.mkPen("g"))
        self._reference_points_item = pg.ScatterPlotItem(size=10, symbol="x", pen=pg.mkPen("r"))
        self._global_planner_segments_items: list[pg.PlotCurveItem] = []
        self._trajectory_item = pg.PlotCurveItem(pen=pg.mkPen("c"))
        self._trajectory_item.setVisible(False)
        self._plot_widget.addItem(self._bounding_box_item)
        self._plot_widget.addItem(self._unknown_obstacles_item)
        self._plot_widget.addItem(self._known_obstacles_item)
        self._plot_widget.addItem(self._trajectory_item)
        self._plot_widget.addItem(self._measured_state_item)
        self._plot_widget.addItem(self._goal_pose_item)
        self._plot_widget.addItem(self._pressed_pose_item)
        self._plot_widget.addItem(self._reference_points_item)
        self._plot_widget.addItem(self._local_trajectory_item)
        self._plot_widget.addItem(self._goal_unreachable_item)
        self._plot_viewbox.disableAutoRange()

        self._velocity_plot_item = pg.PlotCurveItem(pen=pg.mkPen("y"))
        self._steer_plot_item = pg.PlotCurveItem(pen=pg.mkPen("g"))
        self._velocity_plot_widget.addItem(self._velocity_plot_item)
        self._steer_plot_widget.addItem(self._steer_plot_item)

        # declare nodes
        self._map_server_node = MapServerNode()
        self._car_simulation_node = CarSimulationNode(
            delta_time_s=SIMULATION_DELTA_TIME,
            simulation_interval_s=SIMULATION_INTERVAL,
            publish_interval_s=SIMULATION_PUBLISH_INTERVAL,
        )
        self._global_planner_node = GlobalPlannerNode(
            segment_collection_size=GLOBAL_PLANNER_SEGMENT_COLLECTION_SIZE,
        )
        self._local_planner_node = LocalPlannerNode(
            delta_time_s=LOCAL_PLANNER_DELTA_TIME,
            update_interval_s=LOCAL_PLANNER_UPDATE_INTERVAL,
        )
        self._trajectory_collision_checking_node = TrajectoryCollisionCheckingNode()

        # connect signals
        self._car_simulation_node.measured_state.connect(self._local_planner_node.set_state)
        self._car_simulation_node.measured_state.connect(self._map_server_node.update)
        self._car_simulation_node.measured_state.connect(self._update_measured_state)
        self._global_planner_node.display_segments.connect(self._update_global_planner_display_segments)
        self._global_planner_node.finished.connect(self._car_simulation_node.resume)
        self._global_planner_node.trajectory.connect(self._local_planner_node.set_trajectory)
        self._global_planner_node.trajectory.connect(self._trajectory_collision_checking_node.set_trajectory)
        self._global_planner_node.trajectory.connect(self._update_global_planning_result)
        self._local_planner_node.control_sequence.connect(self._car_simulation_node.set_control_sequence)
        self._local_planner_node.local_planning_trajectories.connect(self._update_local_planning_trajectories)
        self._map_server_node.inited.connect(self._inited)
        self._map_server_node.known_obstacle_coordinates_updated.connect(
            self._trajectory_collision_checking_node.set_known_obstacles
        )
        self._map_server_node.known_obstacle_coordinates_updated.connect(self._update_known_obstacle_coordinates)
        self._map_server_node.new_obstacle_coordinates.connect(self._trajectory_collision_checking_node.check_collision)
        self._trajectory_collision_checking_node.collided.connect(self._local_planner_node.brake)
        self._trajectory_collision_checking_node.collided.connect(self._trajectory_collided)
        self.braked.connect(self._global_planner_node.cancel)
        self.braked.connect(self._local_planner_node.brake)
        self.braked.connect(self._trajectory_collision_checking_node.cancel)
        self.canceled.connect(self._car_simulation_node.stop)
        self.canceled.connect(self._global_planner_node.cancel)
        self.canceled.connect(self._local_planner_node.cancel)
        self.canceled.connect(self._trajectory_collision_checking_node.cancel)
        self.restarted.connect(self._map_server_node.init)
        self.set_goal.connect(self._global_planner_node.plan)
        self.set_state.connect(self._car_simulation_node.set_state)

        self._ui.brake_button.clicked.connect(self.brake)
        self._ui.cancel_button.clicked.connect(self.cancel)
        self._ui.restart_button.clicked.connect(self.restart)
        self._ui.set_pose_button.clicked.connect(lambda: self._pressed_pose_item.set_color("w"))
        self._ui.set_goal_button.clicked.connect(lambda: self._pressed_pose_item.set_color("g"))
        self._plot_viewbox.sigMouseDrag.connect(self._mouse_drag)

        # start tasks
        self._map_server_node.init()
        self._car_simulation_node.start()
        self._global_planner_node.start()
        self._local_planner_node.start()

    @Slot()
    def restart(self) -> None:
        self.cancel()
        self._goal_pose_item.setVisible(False)
        self._trajectory_item.setVisible(False)
        self.restarted.emit()

    @Slot()
    def cancel(self) -> None:
        self._goal_unreachable_item.setVisible(False)
        self._clear_global_planner_display_segments()
        self._local_trajectory_item.setData([], [])
        self._reference_points_item.setData([], [])
        self._local_planning = False
        self._brake_trajectory = None
        self.canceled.emit()

    @Slot()
    def brake(self) -> None:
        self._goal_unreachable_item.setVisible(False)
        self._clear_global_planner_display_segments()
        self.braked.emit()

    @Slot(MouseDragEvent)
    def _mouse_drag(self, ev: MouseDragEvent) -> None:
        start_pos = self._plot_viewbox.mapSceneToView(ev.buttonDownScenePos())
        start_x, start_y = start_pos.x(), start_pos.y()
        if not (self._minx <= start_x <= self._maxx and self._miny <= start_y <= self._maxy):
            return

        self._goal_unreachable_item.setVisible(False)

        pos = self._plot_viewbox.mapSceneToView(ev.scenePos())
        x, y = pos.x(), pos.y()
        state = Car(start_x, start_y, np.arctan2(y - start_y, x - start_x))
        self._pressed_pose_item.set_state(state)
        self._pressed_pose_item.setVisible(True)

        if not ev.isFinish():
            return

        self._pressed_pose_item.setVisible(False)
        self._trajectory_item.setVisible(False)

        if self._ui.set_pose_button.isChecked():
            self._goal_pose_item.setVisible(False)
            self.cancel()
            self.set_state.emit(state)
        elif self._ui.set_goal_button.isChecked():
            self._goal_state = state
            self._goal_pose_item.set_state(state)
            self._goal_pose_item.setVisible(True)
            self._goal_unreachable_item.setPos(start_x, start_y)
            self.brake()
            self._global_plan()

    def _global_plan(self) -> None:
        start = self._measured_state
        if abs(start.velocity) > REPLAN_MAX_SPEED and self._brake_trajectory is not None:
            start = self._brake_trajectory
        self.set_goal.emit(start, self._goal_state, Obstacles(self._map_server_node.known_obstacle_coordinates))

    def _inited(self) -> None:
        self._known_obstacles_item.setData(*self._map_server_node.known_obstacle_coordinates.T)
        self._unknown_obstacles_item.setData(*self._map_server_node.unknown_obstacle_coordinates.T)
        self._minx, self._miny, self._maxx, self._maxy = self._map_server_node.bounding_box
        self._plot_viewbox.setXRange(self._minx, self._maxx)
        self._plot_viewbox.setYRange(self._miny, self._maxy)
        self._bounding_box_item.setData(
            [self._minx, self._maxx, self._maxx, self._minx, self._minx],
            [self._miny, self._miny, self._maxy, self._maxy, self._miny],
        )
        self.set_state.emit(self._map_server_node.generate_random_initial_state())

    @Slot()
    def _trajectory_collided(self) -> None:
        self._trajectory_item.setVisible(False)
        self._global_plan()

    @Slot(list)
    def _update_global_planner_display_segments(self, display_segments: list[npt.NDArray[np.floating[Any]]]) -> None:
        connects = [np.ones(len(segment), dtype=bool) for segment in display_segments]
        for i in range(len(connects) - 1):
            connects[i][-1] = False
        connects = np.concatenate(connects)
        segments = np.vstack(display_segments)
        pen = pg.mkPen(len(self._global_planner_segments_items), 16)
        item = pg.PlotCurveItem(*segments.T, pen=pen, skipFiniteCheck=True, connect=connects)
        self._plot_widget.addItem(item)
        self._global_planner_segments_items.append(item)

    def _clear_global_planner_display_segments(self) -> None:
        for item in self._global_planner_segments_items:
            self._plot_widget.removeItem(item)
        self._global_planner_segments_items.clear()

    @Slot(np.ndarray)
    def _update_global_planning_result(self, trajectory: Optional[np.ndarray]) -> None:
        self._clear_global_planner_display_segments()

        if trajectory is not None:
            self._trajectory_item.setData(*trajectory.T[:2])
            self._trajectory_item.setVisible(True)
            self._local_planning = True
        else:
            self._goal_unreachable_item.setVisible(True)

    @Slot(np.ndarray)
    def _update_known_obstacle_coordinates(self, known_obstacle_coordinates: npt.NDArray[np.floating[Any]]) -> None:
        self._known_obstacles_item.setData(*known_obstacle_coordinates.T)

    @Slot(float, Car)
    def _update_measured_state(self, timestamp_s: float, state: Car) -> None:
        self._measured_state = state
        self._measured_timestamp = timestamp_s
        self._measured_timestamps.append(timestamp_s)
        self._measured_velocities.append(state.velocity * 3.6)  # m/s -> km/h
        self._measured_steers.append(np.rad2deg(state.steer))
        timestamps = np.array(self._measured_timestamps)
        velocities = np.array(self._measured_velocities)
        steers = np.array(self._measured_steers)
        self._velocity_plot_item.setData(timestamps, velocities)
        self._steer_plot_item.setData(timestamps, steers)
        self._measured_state_item.set_state(state)
        self._plot_widget.setTitle(f"Timestamp: {timestamp_s:.1f}s")
        self._velocity_plot_widget.setTitle(f"Velocity: {state.velocity * 3.6:.1f}km/h")
        self._steer_plot_widget.setTitle(f"Steer: {np.rad2deg(state.steer):.1f}°")

    @Slot(LocalPlanningTrajectories)
    def _update_local_planning_trajectories(self, local_planning_trajectories: LocalPlanningTrajectories) -> None:
        if self._local_planning:
            self._local_trajectory_item.setData(*local_planning_trajectories.local_trajectory.T[:2])
            self._reference_points_item.setData(*local_planning_trajectories.reference_points.T[:2])
            self._brake_trajectory = local_planning_trajectories.brake_trajectory

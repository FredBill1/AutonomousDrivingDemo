Autonomous Driving Demo
=======================

# Installing Dependencies

Requires Python 3.12 or later.

```bash
pip install -r requirements.txt
```

# Running the Demo

Main Application

```bash
python -m AutonomousDrivingDemo
```

Separate Components

```bash
python -m AutonomousDrivingDemo.demo.car_simulation_and_collision_checking
```

```bash
python -m AutonomousDrivingDemo.demo.global_planning
```

```bash
python -m AutonomousDrivingDemo.demo.local_planning
```

```bash
python -m AutonomousDrivingDemo.demo.global_and_local_planning_combined
```

```bash
python -m AutonomousDrivingDemo.demo.map_server_and_collision_checking
```

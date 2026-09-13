# ros2 topic pub -1 /target/position geometry_msgs/msg/Twist "{linear: {x: 1.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"

## NaVILA deployment

NaVILA is an optional `model_run.py` backend. Place the NaVILA repository and
checkpoint at the paths configured in `configs/chop_navila.yaml`, then run:

```bash
python deployment/model_run.py -m navila
```

Publish the navigation instruction once per task:

```bash
ros2 topic pub --once /goal/text std_msgs/msg/String "{data: 'walk down the hallway and stop near the couch'}"
```

The backend buffers camera frames, runs NaVILA, and adapts its high-level text
command into the existing local `/path` interface. Existing ViNT, GNM, NoMaD,
and OmniVLA model selectors are unchanged.

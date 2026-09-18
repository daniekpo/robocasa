# Collecting task-oracle demonstrations

RoboCasa can execute a short sequence of relational manipulation commands and
store successful rollouts as one local LeRobot v3 dataset per task. The
collector creates a fresh environment for every attempt so stochastic scene
placements vary with the attempt seed.

## Define a task

Task files are YAML and contain between two and five structured commands:

```yaml
name: example_put_objects_in_basket
scene_name: expanded_example_scene
commands:
  - action: move
    object: can
    relation: in
    target: basket
  - action: move
    object: boxed_drink
    relation: in
    target: basket
```

`scene_name` can be a bundled scene name or a path relative to the task file.
The currently supported relations are `in`, `on`, `left_of`, `right_of`,
`in_front_of`, and `behind`. Object and target names must exist in the scene.

## Collect demonstrations

Install the project dependencies, including `lerobot==0.4.4`, then run:

```bash
python -m robocasa.scripts.collect_task_demos \
  --task example_put_objects_in_basket \
  --output outputs/example_put_objects_in_basket \
  --num-demos 10 \
  --seed 0
```

If an existing environment has an older LeRobot release, upgrade that same
Python environment first:

```bash
python -m pip install --upgrade "lerobot==0.4.4"
python -c "import importlib.metadata; print(importlib.metadata.version('lerobot'))"
```

To watch the oracle, add `--view`. The first configured camera is shown by
default; select another with `--view-camera robot_and_counter`. Press `q` in the
window to stop collection. Live viewing requires a graphical display:

```bash
python -m robocasa.scripts.collect_task_demos \
  --task example_put_objects_in_basket \
  --output outputs/example_put_objects_in_basket \
  --num-demos 10 \
  --seed 0 \
  --view \
  --view-camera robot_and_counter
```

Only successful episodes are committed. By default the collector tries at most
three times the requested number of demonstrations; use `--max-attempts` to
change that limit. Failed attempt seeds and errors are appended to
`extras/failures.jsonl`.

RGB observations are video-backed LeRobot features. Depth is a native numeric
`uint16` Array2D feature in millimeters. Each frame also stores the ordered
robot state, ordered object state, flattened simulator state, action, active
command, command index, sparse command-completion reward, and final-task done
flag. The normal LeRobot `task` annotation contains the task name.

Each successful episode has sidecars under `extras/episode_XXXXXX/`:

- the N+1 simulator state sequence and saved model XML;
- camera intrinsics and camera-to-world transforms;
- episode seed and metadata;
- ground-truth and symbolic scene graph traces.

Both graph traces have an initial snapshot and one snapshot after every
successfully verified command. Their `simulator_state_index` fields point into
the saved N+1 state sequence.

## Replay a demonstration

Exact state replay is the default and is appropriate for deterministic visual
inspection:

```bash
python -m robocasa.scripts.replay_task_demo \
  outputs/example_put_objects_in_basket \
  --episode 0 \
  --video outputs/replay.mp4
```

Pass `--actions` to step the recorded actions instead. Action replay reports
the maximum difference from the saved successor states and is useful for
detecting simulator or dependency drift.

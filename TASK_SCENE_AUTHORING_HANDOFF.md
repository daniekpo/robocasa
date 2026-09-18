# Task and scene authoring handoff

The collection pipeline is ready for a larger task/scene suite. Author one
scene YAML for each useful stochastic layout and one or more task YAMLs that
reference it. Reuse a scene when different command sequences should begin from
the same object set and placement distribution.

## Authoring contract

- Choose manipulation objects from `robocasa/global_objects.txt` and confirm
  every chosen category is supported by the scene config loader.
- Give every scene object a unique task-facing `name`.
- Enable RGB-D cameras. The expanded example uses three 640 x 480 views and is
  the reference camera setup.
- Keep every task between two and five commands.
- Use the structured command form below; command-expression strings are not
  accepted.
- Use only `move` with `in`, `on`, `left_of`, `right_of`, `in_front_of`, or
  `behind` for this first suite.
- Ensure each command's `object` and `target` names occur in the referenced
  scene and are different.
- Do not add `open` or `close` commands yet. Those need a corresponding oracle
  implementation and postcondition before entering task files.

```yaml
name: put_objects_in_basket
scene_name: expanded_example_scene
commands:
  - action: move
    object: can
    relation: in
    target: basket
  - action: move
    object: bagel
    relation: in
    target: basket
```

Place bundled files in `robocasa/scene_configs/` and
`robocasa/task_configs/`. Prefer descriptive snake-case task names and stable
object names because they become dataset annotations and scene-graph node IDs.

## Design guidance

Tasks should exercise meaningful long-horizon dependencies rather than repeat
the same independent move. Useful patterns include filling a receptacle before
moving it, stacking before relocating a support, or arranging several objects
relative to a shared landmark. Keep placements reachable by the PandaOmron and
avoid initial contacts unless the relationship is intentional.

The current oracle has a category-specific basket rule that grasps the
robot-nearest rim. Other objects use a center grasp. If a new category cannot
be grasped at its center, record it in the handoff results and add one small
category-specific strategy to `robocasa/oracles/pick_place.py` with a focused
unit test.

## Acceptance checklist

For each proposed task:

1. Load the task with `load_task_config()` and fix all validation errors.
2. Reset its scene over several seeds and visually inspect reachability,
   visibility, collisions, and placement variation.
3. Run the oracle over at least three seeds before requesting a larger
   collection.
4. Confirm every run produces exactly `number_of_commands + 1` snapshots in
   both graph traces.
5. Confirm the symbolic trace applies every command update. Ground-truth
   directional edges are intentionally sparse nearest-peer descriptions, so a
   commanded directional pair can be omitted when it is no longer nearby.
6. Replay one successful episode in state mode and one in action mode.
7. Report success rate and failure reasons by seed. Do not hide failed seeds by
   weakening relation postconditions.

Start from `robocasa/task_configs/example_put_objects_in_basket.yaml` and
`robocasa/scene_configs/expanded_example_scene.yaml`.

The scripted executor remains the baseline for authoring validation. See
`MOTION_PLANNER_HANDOFF.md` before treating its trajectories as expert-quality
demonstrations or adding tasks that require tight collision avoidance.

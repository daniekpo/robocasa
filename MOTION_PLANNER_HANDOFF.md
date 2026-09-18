# Motion-planner integration handoff

The task oracle and data pipeline are working, but the scripted waypoint
executor does not yet produce natural expert demonstrations. Keep the current
executor as a reproducible baseline while adding a collision-aware motion
planner behind a small interface.

## Observed baseline limitations

The behavior is explained by the current constants and waypoint construction
in `robocasa/oracles/pick_place.py`:

- `gripper_settle_steps = 30` holds the end effector stationary for every
  close and open operation. At the current 20 Hz control rate, this creates a
  1.5-second pause after closing the gripper.
- `approach_height = 0.20` raises ordinary objects 20 cm before transport.
- `drop_clearance = 0.20` places an object 20 cm above the top of a receptacle.
- The target transit waypoint adds the approach height above that drop pose.
  For an `in` command, the end effector can therefore travel roughly 40 cm
  above the receptacle rim before descending to release.
- The axis-aligned path is deliberately conservative and has no collision
  model, so reducing these constants alone is not a robust solution.

These motions are useful for validating task semantics and recording, but the
resulting trajectories should not be treated as human-expert demonstrations.

## Goal and scope

Replace fixed axis-aligned transport with collision-aware, closed-loop motion
that preserves the existing task-level oracle, relation postconditions,
transition callbacks, dataset fields, and scene-graph command boundaries.

This is a geometric motion-planning task. Do not add a language or vision
planner. The existing oracle should continue to choose the source, target,
relation, grasp point, and placement goal. The new component should plan and
execute robot motion between those goals.

## Recommended boundary

Introduce one narrow planner contract rather than rewriting `TaskOracle`.
Conceptually, it should accept:

- the current robot configuration and end-effector pose;
- a target end-effector pose or grasp/place goal;
- the static and movable collision geometry from the current MuJoCo scene;
- the held object's geometry and grasp transform after pickup; and
- planning tolerances and a time budget.

It should return a time-parameterized joint-space trajectory, or a clear
failure result with diagnostics. A motion executor should track that
trajectory through the existing environment step path so every transition is
still recorded. It should replan or fail explicitly when tracking error,
unexpected contact, or object slip invalidates the plan.

Keep grasp closure separate from geometric planning. Replace the blind
30-step hold with an observable completion rule using gripper position or
width convergence and object contact/grasp state, with a short maximum
timeout. Start lifting as soon as grasp closure is established. Opening can
use the same convergence rule.

## Planning sequence

For each move command, plan and execute these phases:

1. Approach a pre-grasp pose without collision.
2. Follow a short constrained path to the grasp pose.
3. Close until contact or gripper convergence confirms the grasp.
4. Lift only enough to clear the source and nearby geometry.
5. Plan transport with the grasped object included in collision checking.
6. Approach a pre-place pose and descend along the placement constraint.
7. Release close to the final support surface or receptacle interior.
8. Retreat without disturbing the placed object.

For receptacle placement, compute release height from source and target
geometry. Begin evaluation with 1--3 cm of vertical clearance, subject to the
planner's collision result, instead of the current fixed 20 cm. A loaded
basket must be represented as a compound carried object so its contents and
rim participate in collision checking.

## Planner evaluation

Evaluate candidate backends before committing the oracle to one. At minimum,
check:

- compatibility with Python 3.10, MuJoCo, PandaOmron, and the existing
  robosuite controller stack;
- support for joint limits, self-collision, scene meshes, and attached-object
  collision geometry;
- deterministic seeding and useful failure diagnostics;
- CPU/GPU requirements, installation burden, and license; and
- whether it performs global collision-aware planning rather than only local
  inverse kinematics.

An IK package can still be used for constrained approach and retreat, but it
does not by itself replace collision-aware transport planning. Compare a
MuJoCo/OMPL integration and a GPU planner such as cuRobo only after confirming
the requirements above; do not add both backends preemptively.

## Incremental implementation

1. Add gripper completion sensing and tests while retaining scripted motion.
2. Add the planner interface and construct a collision world from a reset
   configured scene.
3. Use planned motion for unladen approach, lift, transport, placement, and
   retreat while keeping the current grasp strategies.
4. Add attached-object collision checking and validate ordinary objects.
5. Validate the basket edge grasp and transport a loaded basket. Treat this as
   a separate milestone because it was not reliable with the scripted oracle.

Do not delete the scripted executor until the planned path passes the same
tasks over multiple stochastic resets. A runtime choice between the two is
useful during evaluation, but avoid a larger plugin framework until a second
backend is actually needed.

## Acceptance criteria

Use the same scene seeds to compare scripted and planned execution. Record at
least success, episode duration, end-effector path length, maximum transport
height, release height above the target, collision count, planning time, and
replan count.

The planned executor is ready for collection when:

- there is no stationary post-close pause longer than 0.25 seconds after the
  grasp has been confirmed;
- ordinary receptacle placements release 1--3 cm above the valid placement
  surface unless collision geometry requires more clearance;
- paths have no robot, carried-object, or environment collisions other than
  intended grasp/place contact;
- each command still emits transitions through the existing callback and
  produces exactly one verified command boundary;
- state replay remains deterministic and action replay remains compatible;
- two-to-five-command tasks succeed across the agreed seed set without
  weakening relation postconditions; and
- planner failures reject the rollout with actionable diagnostics rather than
  recording a partial successful demo.

## Suggested first experiment

Use `example_put_objects_in_basket` and run both executors on the same ten
seeds. Plot end-effector height and gripper command over time, then inspect the
maximum height above the basket rim and the interval from confirmed grasp to
the first upward motion. This directly measures the two artifacts that
motivated the planner work before expanding to harder tasks.

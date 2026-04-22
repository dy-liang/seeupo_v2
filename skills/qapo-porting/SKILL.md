---
name: qapo-porting
description: Use this skill when porting the QAPO-style multi-turn RL modification into another codebase. It covers reward semantics, rollout/termination assumptions, reward metadata, QAPO advantage computation, config switches, and migration checks for GRPO-style trainers.
---

# QAPO Porting

Use this skill when adding the QAPO variant from this project into another multi-turn RL codebase.

This skill is for code migration, not paper writing. Focus on preserving semantics and tensor shapes.

## QAPO Goal

QAPO adds a separate advantage-estimator branch with two group-relative advantages:

- `A_step`: group-relative advantage from trajectory-level normalized reward
- `A_token`: group-relative advantage from token-length normalized reward
- final advantage: `A = A_step + beta * A_token`

In this project, QAPO is enabled by:

- `adv_estimator: qapo`

Do not hide QAPO behind generic GRPO logic. Keep it as an explicit branch.

## Current Project Semantics

The codebase currently uses these meanings. Preserve them unless you intentionally want a different algorithm.

### Reward source

- The main training reward is trajectory-level, not step-level.
- Reward is computed after rollout ends.
- In AppWorld, rollout ends when the agent explicitly calls `complete_task(...)` or other external stop conditions happen.
- Final environment score comes from evaluator-based requirement checks, not from counting successful interaction steps.

### Important distinction

- `step` != `requirement`
- `num_passes` / `num_failures` are counts over evaluator requirements, not trajectory steps.
- Do not use `num_passes` as a step count unless you explicitly want requirement-based normalization.

### Current QAPO reward semantics in this repo

- Step-normalized branch:
  - if `active_step == False`, reward is `0.0`
  - if `success_rate > 0`, use `outcome / trajectory_length`
  - otherwise use raw `outcome`
- Token-normalized branch:
  - if `active_step == False`, reward is `0.0`
  - if `outcome > 0`, use `outcome / effective_response_token_count`
  - otherwise use raw `outcome`
- `trajectory_length` means real trajectory step count, not padded turns and not requirement count.
- `effective_response_token_count` must come from the same mask used for the actor loss, typically `loss_mask` on the response span.

If you want a stricter QAPO variant, change both branches together and document the rule.

## Files/Areas To Port

When migrating QAPO, usually touch these four areas:

1. Reward production at rollout end
2. Per-step sample metadata
3. Advantage computation branch
4. Config and launcher wiring

## Migration Checklist

### 1. Reward production

At rollout end, make sure each trajectory has:

- `reward.outcome`
- `reward.success_rate`
- optional `reward.metadata`

If your environment has evaluator details, store them in metadata, but do not confuse them with steps.

### 2. Per-step reward metadata

When turning one trajectory into per-step samples, each step sample should carry:

- `outcome`
- `success_rate`
- `trajectory_length`
- `active_step`

Rules:

- `trajectory_length` = real number of interaction steps in the trajectory
- padded turns must have `active_step=False`
- padded turns must not affect reward or group statistics

### 3. Reward tensor path

Keep base reward parsing simple:

- for standard GRPO-like training, parse raw outcome into the reward tensor
- usually place the scalar reward on the last valid response token
- let QAPO-specific normalization happen inside the QAPO advantage function, not in the generic reward parser

This avoids leaking QAPO behavior into GRPO or other algorithms.

### 4. Add explicit QAPO estimator

Add:

- `AdvantageEstimator.QAPO = "qapo"`

Then create:

- `elif adv_estimator == AdvantageEstimator.QAPO`
- `compute_qapo_outcome_advantage(...)`

Do not overload the GRPO branch to silently behave like QAPO.

### 5. QAPO computation shape contract

Input:

- `reward_scores`: length `bs`, object array / list of dicts
- `response_mask`: shape `(bs, response_len)`
- `index`: grouping ids, length `bs`
- `response_token_lengths`: shape `(bs,)`

Output:

- `advantages`: shape `(bs, response_len)`
- `returns`: shape `(bs, response_len)`

Recommended implementation:

1. Build scalar `trajectory_scores` from step-normalized reward
2. Compute group-relative advantages from `trajectory_scores`
3. Build scalar `token_scores` from token-normalized reward
4. Compute group-relative advantages from `token_scores`
5. Combine:
   `combined = traj_adv + beta * token_adv`

Use the same group-relative helper for both branches.

## Required Helper Behavior

Your group-relative helper should:

- group samples by `index`
- compute group mean and std on active samples only
- set inactive padded samples to zero
- expand scalar group-relative scores back to `(bs, response_len)` using the response/loss mask

This is the core invariant of QAPO.

## Config Interface

Minimum config:

```yaml
algorithm:
  adv_estimator: qapo
  token_level_adv_beta: 1.0
```

Recommended:

- keep `grpo` and `qapo` as separate values
- set `token_level_adv_beta: 0.0` in non-QAPO launchers for clarity

## Reward Shaping Convention In This Repo

This repo also changes the success bonus when `adv_estimator == qapo`:

- QAPO success reward bonus: `10.0 + score * 0.5`
- non-QAPO success reward bonus: `1.0 + score * 0.5`

Port this only if you want the same training behavior. It is not required by QAPO mathematically.

## Metrics To Keep

Useful rollout metrics:

- successful trajectory count
- successful trajectory step length avg/max/min
- partial-success trajectory step length avg/max/min
- failure trajectory step length avg/max/min
- complete-success `num_passes` avg/max/min
- partial-success `num_passes` avg/max/min
- total requirement count `num_passes + num_failures` avg/max/min

These are diagnostics only. They should not affect training logic.

## Pitfalls

- Do not treat evaluator requirement count as step count.
- Do not let padded turns enter group mean/std.
- Do not normalize inside the generic GRPO reward path if only QAPO should change.
- Do not assume rollout termination means evaluator success.
- Do not assume final evaluator success requires every intermediate step to be locally correct.

## Quick Validation

After porting, verify:

1. `grpo` still runs without entering QAPO code.
2. `qapo` uses the dedicated advantage branch.
3. Padded turns get zero reward and zero advantage.
4. `trajectory_length` equals real interaction steps.
5. `response_token_lengths` comes from the same effective response mask used for actor loss.
6. `advantages.shape == returns.shape == (bs, response_len)`.
7. Switching `adv_estimator` between `grpo` and `qapo` changes only the intended logic.

## Minimal Porting Recipe

When migrating to a new project:

1. Add `qapo` to the advantage estimator enum.
2. Ensure rollout trajectories expose `outcome`, `success_rate`, real `trajectory_length`, and `active_step`.
3. Keep base reward parsing algorithm-agnostic.
4. Add `compute_qapo_outcome_advantage(...)`.
5. Use one shared group-relative helper for both scalar score branches.
6. Add `token_level_adv_beta` to config.
7. Add a dedicated QAPO launcher or config preset.

## What To Re-check Per New Codebase

- Does the environment reward mean final-task success, partial evaluator score, or dense step reward?
- Does rollout termination mean solved, failed, or merely agent-declared done?
- Are step samples true turns or arbitrary chunks?
- Which mask is the true effective response mask for training?
- Is there any existing reward shaping that would accidentally stack with QAPO?

If any of these answers differ, adapt the QAPO port instead of copying it blindly.

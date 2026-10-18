# Switchback roadmap

The next phase focuses on creative tools, replayable challenges, and more ways
to play. Estimates are planning ranges for unfinished work and will change as
the features are tested. Completed v0.2 features are listed separately below.

| Priority | Deliverable | Estimate | Acceptance criteria |
| --- | --- | --- | --- |
| 1 | Local level editor and validation | 6–8 h | Place walls, gates, starts, and energy targets; undo/redo; preview a stage; import/export versioned JSON. Reject malformed, overlapping, out-of-bounds, and oversized input. Keep the authored campaign intact. |
| 2 | Replay recorder and viewer | 6–8 h | Record tick-indexed turns/shifts, stage or seed, and engine version. Export/import bounded JSON; pause and change playback speed; reproduce results exactly and reject incompatible versions. |
| 3 | Configurable input and accessibility | 4–6 h | Remap controls, add left/right-handed touch layouts and gamepad input/disconnect handling, and verify focus/live-region behavior with assistive technology. Build on existing touch, swipe, symbols, and high contrast. |
| 4 | Offline installation and update handling | 4–6 h | Add install metadata and a versioned offline cache; handle interrupted upgrades and stale saves; test on actual phones/tablets and desktop browsers. Build on the existing static release. |
| 5 | Bonus-stage mechanics and campaign balancing | 4–6 h | Add switch-activated gates and limited-shift objectives, at least three bonus stages, rule tests, and calibrated medal targets from playtests. |
| | **Total estimated work** | **24–34 h** | Verify each feature through working behavior, focused tests, and release notes. |

## Available now

- Six authored stages with distinct layouts and verified completion routes.
- Guided First Light lessons, stage unlocks, saved progress, and route medals.
- Phase run, Classic, and UTC daily arcade modes; local scores and adjustable pace.
- Keyboard, touch buttons, swipe steering, pause/retry, high contrast, and optional
  original synthesized sound.
- Backward-compatible saves and separate recent-history/campaign reset controls.
- Dependency-free development server, public-asset build, 42 tests, CI, and release docs.

## Design principles

Keep the game playable locally without accounts or external services. Replay
tools should help players practice routes and reproduce bugs. Custom stages and
replays should use bounded, validated files. Preserve existing saves, campaign
progress, and the small static distribution as new features arrive.

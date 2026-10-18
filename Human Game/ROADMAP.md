# Work remaining after v0.2.0

The owner delegated feature decisions and confirmed an intent to continue
Switchback for at least 20 hours regardless of CC-CLI v2 acceptance. The selected
next phase is estimated at **24–34 hours**. These estimates describe unfinished
work, not time already worked or a program approval. Re-estimate as work progresses.

| Priority | Deliverable | Estimate | Acceptance criteria |
| --- | --- | --- | --- |
| 1 | Local level editor and validation | 6–8 h | Place walls, gates, starts, and energy targets; undo/redo; preview a stage; import/export versioned JSON. Reject malformed, overlapping, out-of-bounds, and oversized input. Keep the authored campaign intact. |
| 2 | Human-run replay recorder and viewer | 6–8 h | Record tick-indexed turns/shifts, stage or seed, and engine version. Export/import bounded JSON; pause and change playback speed; reproduce results exactly and reject incompatible versions. No learned player or model evaluation. |
| 3 | Configurable input and accessibility | 4–6 h | Remap controls, add left/right-handed touch layouts and gamepad input/disconnect handling, and verify focus/live-region behavior with assistive technology. Basic touch, swipe, symbols, and high contrast already exist and are excluded. |
| 4 | Offline installation and update handling | 4–6 h | Add install metadata and a versioned offline cache; handle interrupted upgrades and stale saves; test on actual phones/tablets and desktop browsers. The static build is already complete and is excluded. |
| 5 | Bonus-stage mechanics and campaign balancing | 4–6 h | Add switch-activated gates and limited-shift objectives, at least three bonus stages, rule tests, and calibrated medal targets from the owner's playtests. Existing six stages, lessons, sound, and medals are excluded. |
| | **Total unfinished work** | **24–34 h** | Show completion through code, behavior checks, and release notes. |

## Already delivered; excluded from the estimate

- Six authored stages with distinct layouts and verified completion routes.
- Guided First Light lessons, stage unlocks, saved progress, and route medals.
- Phase run, Classic, and UTC daily arcade modes; local scores and adjustable pace.
- Keyboard, touch buttons, swipe steering, pause/retry, high contrast, and optional
  original synthesized sound.
- Backward-compatible saves and separate recent-history/campaign reset controls.
- Dependency-free development server, public-asset build, 42 tests, CI, and release docs.
- Setup, architecture, ownership/eligibility evidence, and submission draft.

## Boundaries

This is a human-played game. The future work includes no model development,
training, evaluation, inference, ML infrastructure, or data creation for those uses.
Replay tools reproduce human input for practice and debugging. No personal or
customer data is needed. Program acceptance remains the organizers' decision.

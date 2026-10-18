# Work remaining after v0.1.0

This is the selected **30–42 hour** next-phase backlog for a human-played game.
On 2026-10-17, the owner delegated feature selection to the coding assistant;
the milestones below prioritize interesting gameplay and practical game tools.
None is implemented in the starter release. Before applying, the owner must
confirm that this is work they intend to do regardless of program participation.
Re-estimate after discovery; do not claim these estimates as time already worked.

| Priority | Deliverable | Estimate | Acceptance criteria |
| --- | --- | --- | --- |
| 1 | Authored campaign and progression | 6–8 h | At least six distinct, completable stages with objectives, stage selection, progression saves, a win screen, and migration tests. Stages must offer different decisions rather than palette swaps. |
| 2 | Local level editor and validation | 6–8 h | Place walls, gates, start positions, and objectives; undo/redo; preview a stage; import/export a versioned JSON format. Reject malformed, out-of-bounds, overlapping, or oversized levels without breaking the game. |
| 3 | Deterministic replay recorder and viewer | 6–8 h | Record tick-indexed turns and shifts with engine version and seed; export/import a bounded JSON file; pause and change playback speed; reproduce a run exactly. No trained agent or model comparisons. |
| 4 | Configurable controls and accessibility | 4–6 h | Remap keys, add left/right handed touch layouts and gamepad input, handle disconnects, provide a slow practice mode, and verify focus order, reduced motion, and gate symbols. Document canvas accessibility limitations. |
| 5 | Offline installation and device validation | 4–6 h | Add a manifest and versioned offline cache; exercise update behavior; test desktop and actual touch devices; document installation and recovery. No accounts or online scores. |
| 6 | Gameplay polish and release preparation | 4–6 h | Add original optional sound effects with mute; improve feedback and stage pacing through the owner's playtests; resolve reproducible bugs; publish a changelog and release checklist. Store no other person's personal data. |
| | **Total proposed work remaining** | **30–42 h** | Completion is demonstrated by code, tests, and clear release notes. |

## Already implemented; excluded from the remaining estimate

- Human-controlled Phase run, Classic, and UTC daily seed modes.
- Two-phase gate and food rules; safe turn buffering; pause and retry.
- Keyboard and basic touch controls, local records, and a high-contrast option.
- Dependency-free development server, automated rules/storage checks, and CI.
- Setup instructions, architecture, goal, and submission planning documents.

## Boundaries

The proposed work does not include neural networks, RL, model training, model
evaluation, model inference, ML infrastructure, or data creation for those uses.
Future replays are for reproducing human gameplay and debugging game rules.
There is no promise that the program will approve this repository or backlog.

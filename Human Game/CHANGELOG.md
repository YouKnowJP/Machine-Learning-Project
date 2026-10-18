# Changelog

## 0.2.1 — 2026-10-18

- Reorganized the README, project goals, roadmap, QA record, and release guide
  around playing and developing Switchback.
- Simplified the downloadable package's quick-start instructions and linked its
  versioned source.
- Checked that the package version and displayed game version agree.

## 0.2.0 — 2026-10-18

- Added six authored campaign stages, fixed energy objectives, ordered unlocks,
  saved progress, and Gold/Silver/Bronze medals based on move count.
- Added First Light's three playable lessons, automatic instruction pauses,
  slower teaching pace, upcoming-energy markers, and next-stage flow.
- Added optional original synthesized sound, touch swipe steering, and separate
  recent-history clearing and confirmed campaign reset actions.
- Preserved v0.1 scores/preferences when loading upgraded saves.
- Added a public-asset static build with safe output replacement, release docs,
  and build verification in CI. Expanded the automated suite from 29 to 42 tests.
- Verified all six completion routes through the browser interface.
- Updated the roadmap for the level editor, replays, input configuration, offline
  installation, and bonus stages.

## Repository maintenance — 2026-10-17

- Removed the restricted bundled Arial font from the older Snake project and
  selected Pygame's default font there. Historical copies are disclosed in the
  repository's third-party notice.

## 0.1.0 — 2026-10-17

- Added the human-controlled, two-phase Snake game with Classic and UTC daily modes.
- Added keyboard/basic touch controls, local records, high contrast, and pause/retry.
- Added a dependency-free local server, 29 regression tests, and GitHub Actions.
- Added original interface artwork, setup instructions, architecture, a screenshot,
  and a gameplay roadmap.

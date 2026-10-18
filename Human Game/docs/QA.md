# Verification record

## Automated checks

The v0.2.1 release uses the built-in Node test runner:

```sh
npm run check
npm test
npm run build
```

The suite contains **42 tests**. Syntax checks run on the game modules, builder,
and development server. GitHub Actions runs checks, tests, and a build on Node.js
22 on Ubuntu; local verification uses Node.js 26.0.0.

Coverage includes safe turn buffering, opposite-direction rejection, matching
energy, growth, gates, phase shifts, collisions, moving into a vacating tail,
pause behavior, full-board completion, deterministic seeds, food placement over
500 seeds, UTC daily seeds, score separation, and speed bounds. Campaign tests
cover all six executable completion routes, stage integrity, unlock ordering,
and move-based medals. Persistence tests cover corrupted/blocked storage,
score retention, settings, progress, history clearing, and v0.1 save migration.

Build tests check that all relative module imports resolve, the license is
included, the displayed version matches the package, development files are
excluded, stale output is removed, and unrelated nonempty directories are protected.

## Campaign and browser checks

These Chromium checks were completed for v0.2.0. The v0.2.1 update changes
documentation, packaging text, and the displayed version; gameplay is unchanged.

- Every campaign stage was completed through browser input fixtures: First Light
  11 moves, Crosscurrent 37, Slalom 57, Double Lock 67, Orbit 45, Afterglow 73.
  Each awarded Gold; all six completions and unlocks survived reload.
- First Light paused at its two instruction milestones, resumed correctly,
  completed through its mint gate, unlocked Crosscurrent, and advanced with
  Next stage. Reload recommended the first unfinished stage.
- Phase run scored a pickup; Space changed phase; pause froze movement; resume
  allowed a later boundary crash; the score survived reload.
- Daily mode locked its date seed and Standard pace. Classic had no phase-gate
  requirement. Window blur paused play. High contrast survived reload.
- Button clicks enabled and muted synthesized sound without page errors.
- At 390×844, touch buttons and a pointer swipe steered the snake, pause worked,
  and the page had no horizontal overflow. This was browser emulation.
- At 1366×768, the active campaign board fit inside the viewport. Desktop and
  narrow-width screenshots were visually inspected.
- Clearing recent history preserved campaign progress. Cancelling the native
  campaign reset dialog preserved it; accepting cleared completions and relocked stages.
- With browser storage blocked, the game loaded without page errors and reported
  that results could only be retained for the current visit.
- The development server returned 200 for game modules, 404 for documents, and
  405 for POST requests.

Scripted route fixtures live only in tests and are excluded from the static
package. The shipped game is controlled by the player.

## Security and asset checks

Gitleaks 8.30.1 reported no leaks in earlier source-tree and fetched-history
scans. These are pattern-based checks, not a guarantee about every file or ref.
Switchback bundles no external fonts, audio files, models, or runtime packages.
See the root [third-party notice](../../THIRD_PARTY_NOTICES.md) for the older
Snake project's font replacement and historical asset details.

## Remaining verification

- Firefox and Safari.
- Actual phone/tablet play, rather than touch emulation.
- Gamepad input when implemented.
- Screen-reader behavior and a full accessibility review.
- Offline installation and cache upgrades when implemented.

Local scores remain editable by the browser owner. There is no online leaderboard.

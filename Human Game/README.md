# Switchback

**Six circuits. Endless detours.**

Switchback is a human-played Snake arcade game for short skill challenges.
Switch between mint circles and amber diamonds to collect energy and pass
matching gates. Learn the rules through a playable lesson, complete six authored
stages, or chase a high score in three arcade modes.

Built with vanilla JavaScript, Canvas 2D, HTML, and CSS. There are no runtime
packages, models, trained agents, backend accounts, telemetry, or external asset
downloads. Gameplay, progress, and sound synthesis run in the browser.

![Switchback campaign gameplay](docs/switchback.png)

## Run

Install Node.js 22 or newer. From the repository root:

```sh
cd 'Human Game'
npm start
```

Open **http://127.0.0.1:4173**. No package installation or build is needed.
Stop the server with Ctrl+C. If the port is busy, run `PORT=4174 npm start` on
macOS/Linux or `$env:PORT=4174; npm start` in PowerShell.

Alternatively, run `python3 -m http.server 4173 --bind 127.0.0.1` from this folder.
Use an HTTP server: opening `index.html` through `file://` does not reliably load
JavaScript modules. The bundled development server exposes only game assets.

## Four ways to play

| Mode | Challenge |
| --- | --- |
| **Campaign** | Six ordered stages, fixed energy positions, unlocks, and saved route medals. First Light teaches the rules through three playable lessons. |
| **Phase run** | An endless seeded arena. Match your head to energy and gates while leaving room for your tail. |
| **Classic** | An open board with no phase requirements, gates, or solid blocks. |
| **Daily circuit** | A UTC date seed and Standard pace. Unlimited local retries; a shared starting condition without an online leaderboard. |

### Campaign

| Stage | What changes | Energy target | Gold move target |
| --- | --- | --- | --- |
| First Light | Learn movement, switching, and a first gate | 3 | 14 |
| Crosscurrent | Find openings through one divider | 4 | 48 |
| Slalom | Weave through three offset lanes | 5 | 72 |
| Double Lock | Switch between two opposing gate lines | 6 | 86 |
| Orbit | Enter and escape two nested rings | 6 | 60 |
| Afterglow | Connect all four sections of a final crossing | 7 | 94 |

Collect every energy cell to complete a stage and unlock the next. Gold requires
finishing within the listed move target, Silver within 1.5× that target, and
Bronze rewards any completion. Medals use simulation moves, so pace and time spent
paused do not affect the rating. Replaying a stage preserves your best medal and
lowest move count. Any pace can unlock stages; these are local practice records.

First Light uses a slower 220 ms tick. Before its first completion, the game
pauses after the first two pickups so you can read and continue each lesson.
Small numbered rings show later energy positions. The current energy remains
visible even if your tail temporarily covers its cell.

## Controls and rules

| Action | Keyboard | Touch / pointer |
| --- | --- | --- |
| Start, continue, retry, next stage | Tab to the main button, then Enter | Main board button |
| Turn | Arrow keys or WASD | On-screen arrows or swipe the board |
| Switch mint / amber | Space | Shift phase |
| Pause / resume | P or Escape | Pause / Keep going |
| Reset current run | R | New run |
| Sound / high contrast | Tab to the footer controls | Footer toggles |

- Each matching energy pickup is **100 points** and grows the snake by one.
- Match the **entering head** to the gate. Your body can trail through it after a
  phase change. Solid blocks never allow passage.
- Unmatched energy stays available when you pass over it. Classic skips matching.
- Board edges, solid blocks, your body, or a wrong-phase gate end the run. The
  board does not wrap; moving into a tail square that vacates on the same step is legal.
- The pace gently increases every three pickups, except during First Light.
- Switching tabs/windows pauses play. Resume deliberately when you return.
- Sound starts **off**. Optional effects are original synthesized tones with no
  downloaded audio files. A sound failure does not prevent play.
- Route, stage, and pace settings are locked during a live or paused run. Reset
  the run to change them.

## Saved progress and privacy

The browser saves campaign progress, medals, per-mode/pace bests, settings, and
the last five completed runs in `localStorage`. Daily bests are separated by date;
campaign bests are also separated by stage. Upgrading from v0.1 retains existing
arcade scores and settings.

**Clear history** clears recent runs while preserving bests, settings, and campaign
progress. **Reset** in the campaign panel asks before removing campaign unlocks,
medals, and campaign best scores. Clearing browser site data removes everything.
If storage is blocked, the game keeps results for the current visit and reports
that they could not be saved.

There are no names, logins, cookies, telemetry, uploads, remote APIs, or online
score submissions. Local records can be edited by the browser owner; they are not
a trusted competition system. Campaign energy positions are authored. Arcade
seeds repeat only for the same mode, pace, engine version, and inputs; later food
can differ when players take different routes.

## Check and package

```sh
npm run check
npm test
npm run build
```

The **42 tests** cover game rules, all six executable completion routes,
progression, medals, storage migration/corruption, and static release integrity.
GitHub Actions checks syntax, runs tests, and builds the static release.
See [QA notes](docs/QA.md) for browser coverage and limits.

The build writes only public game assets and the MIT license into `dist/`.
Serve that directory with a static host or an HTTP server. It contains no legacy
ML files, developer server, development documentation, or tests. No hosting account or
deployment is required to play locally. The build refuses to overwrite an
unrelated nonempty directory. See [release instructions](docs/RELEASING.md).

```text
src/engine.mjs       Deterministic movement, collisions, energy, and game states
src/levels.mjs       Six authored stages, unlock order, and medal rules
src/storage.mjs      Backward-compatible, validated local saves
src/audio.mjs        Optional original sound synthesis
src/app.mjs          Canvas renderer, lessons, input, and interface
scripts/build.mjs   Static release packaging
tests/              Rules, persistence, campaign routes, and build verification
```

## License and development

MIT; see the included [LICENSE](LICENSE), matching the repository's root license.
Switchback is a personal project developed with AI coding assistance. Its source,
interface artwork, and synthesized sound were created for this game.

Version **0.2.1** includes the six-stage campaign and three arcade modes.
Download the [latest release](https://github.com/YouKnowJP/Machine-Learning-Project/releases/latest)
or run from source. Next features include a level editor, replay tools,
configurable controls/gamepad support, offline installation, and bonus stages;
see the [roadmap](ROADMAP.md) and [project goals](docs/PROJECT_GOAL.md).

See the root [third-party notice](../THIRD_PARTY_NOTICES.md) for asset and
dependency information about the wider repository.

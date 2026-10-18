# Architecture

The game is a buildless static application. `serve.mjs` provides a development
HTTP origin on loopback; no game backend exists. HTML, CSS, modules, and the SVG
icon can also be served by an ordinary static host.

## Rules

`engine.mjs` owns all simulation state. `createGame` initializes a 24×24 board,
snake, seed, arena, and food. A stored xorshift32 state supplies deterministic
food placement; no randomness is used by movement. `step` advances exactly one
cell. A single pending turn prevents two turns from reversing a snake within one
tick. Opposite and duplicate-direction turns do not consume the buffer.

An eating step keeps the tail; a non-eating step removes it. Collision checks
therefore omit the tail only for non-eating moves. Food spawning enumerates empty
cells instead of retrying random cells forever. No empty spawn cell means a win.
The engine has explicit ready, running, paused, over, and won states. Input and
simulation are ignored outside the appropriate states.

Gates check only the entering head's phase. Food in Phase run and Daily must
match the head; Classic skips matching. The arena has gaps between dividers.
Scoring is 100 points per energy, without wall-clock bonuses. Pace depends only
on selected difficulty and number collected. Daily forces Standard pace.

## Browser adapter

`app.mjs` maps keyboard and pointer events to engine calls. A requestAnimationFrame
loop uses a fixed-step accumulator and clamps long frame gaps to 250 ms, avoiding
an unbounded catch-up loop. Page hiding/window blur explicitly pauses play. A
small Canvas renderer draws the grid, walls, phase symbols, food, and snake.
Reduced-motion preference removes pulsing and flashes.

Controls, scores, instructions, pause state, and results are DOM elements.
Important events also update a polite live region. This is not a fully nonvisual
game: the board's spatial state is still drawn on Canvas. A high-contrast palette
and distinct circle/diamond symbols aid visibility; full accessibility and
broader device verification remain future work.

## Persistence

`storage.mjs` reads and validates a namespaced localStorage record. It stores
settings, per-mode/pace best scores, date-keyed daily bests, and five recent runs.
It stores no names, credentials, device identifiers, or remote analytics.
Storage exceptions return a safe fallback and are surfaced by the browser UI.
Run histories are displayed using text nodes, never interpreted as HTML.

Local scores are editable by the browser's owner. They are practice records,
not a trusted competition system. Seed equality guarantees repeatability only
for equal engine version, options, and input sequence. Replays and a versioned
level format are planned, not implemented.

## Isolation

This component imports no code from `Game AI` or other ML directories. The
runtime has no third-party dependencies or network API requests. CI is scoped
to this directory and its workflow. Existing ML files and history are preserved.

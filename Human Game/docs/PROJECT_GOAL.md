# Project goals

Switchback turns Snake into a short routing challenge: switch phases, time a
gate crossing, and find a better path through a circuit. The goal is a polished
single-player game that is easy to start and rewarding to replay.

## Player experience

- Teach movement, phase switching, and gates through play.
- Offer a clear progression from the first lesson to six distinct campaign stages.
- Reward efficient routes with medals while allowing unlimited retries.
- Provide quick arcade sessions through Phase run, Classic, and Daily circuit.
- Support keyboard and touch input, readable symbols, high contrast, and optional sound.

## Engineering goals

- Keep the browser game independent of the repository's other projects.
- Run locally or on a static host without accounts or a backend.
- Keep movement and scoring deterministic for repeatable rule tests and replays.
- Validate saves and retain older scores as the game evolves.
- Ship only public game assets and the MIT license in downloadable packages.
- Document controls, limitations, architecture, and release steps.

## Development status

The six-stage campaign, lessons, unlocks, medals, arcade modes, local saves,
swipe controls, synthesized sound, and static packaging are available now.

Next milestones are a level editor, replay tools, configurable controls and
gamepad support, offline installation, and bonus stages. See the
[roadmap](../ROADMAP.md) for estimates and completion criteria.

Full Firefox/Safari, actual touch-device, and assistive-technology verification
remain open. See the [QA record](QA.md) for tested behavior and current limits.

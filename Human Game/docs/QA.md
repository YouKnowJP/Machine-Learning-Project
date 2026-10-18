# Verification record

Current release: **v0.2.0**, with 42 passing automated tests. Current campaign
and browser results are recorded in the v0.2.0 section below. Earlier checks are
retained as historical evidence for the original release and font correction.

## Original v0.1.0 checks

Checked on **2026-10-17 (America/Los_Angeles)** in the development workspace.

## Automated checks

- `npm run check`: passed on Node.js 26.0.0.
- `npm test`: **29 passed, 0 failed**, using the built-in Node test runner.
- The CI workflow targets Node.js 22 on Ubuntu, with GitHub-maintained actions
  pinned to specific commits. A local pass does not assert a remote CI result.
- `git diff --check`: passed.

Rules coverage includes safe turn buffering, opposite-direction rejection,
matched/unmatched energy, growth, gates, shifting, walls/edges/body collisions,
moving into a vacating tail, pause input, stable finished states, full-board
completion, deterministic input sequences, food placement over 500 seeds, UTC
daily seeds, score separation, and speed bounds. Persistence tests cover corrupt
and blocked storage, preferences, score history, high-score retention, and clearing.

## Browser integration

Playwright Chromium checks passed for:

1. Seed `qa-1802`: start a Phase run, advance one step, and verify 100 points.
2. Keyboard Space changes the phase to amber.
3. Pause freezes a live game; resume permits movement and a later boundary crash.
4. The final score appears in local records and survives page reload.
5. Daily mode locks its date seed and standard pace.
6. Classic mode changes its objective and has no phase-gate requirement.
7. Window blur automatically pauses the run.
8. High contrast changes the palette and survives reload.
9. The development server returns 200 for game modules, 404 for documents, and
   405 for a POST request.
10. At 390×844, actual button clicks turn, shift, and pause; touch controls are
    visible and no horizontal page overflow occurs.
11. At 1440×1120, the desktop layout has no horizontal overflow. Desktop and
    narrow-width screenshots were visually inspected. No page errors occurred
    during the narrow-width interaction check.
12. At 1366×768, starting a run scrolls the game into view and the entire board
    fits inside the viewport. Desktop board sizing follows the available height.

Fixed-clock browser checks dispatch DOM input events; separate pointer and actual
keyboard checks exercise the rendered controls. Browser automation is validation
of a human-controlled game, not an automated player shipped in the application.

## Secrets check

Gitleaks **8.30.1**, downloaded from its official release and checked against its
published SHA-256 checksum, reported **no leaks** for:

```sh
gitleaks git --redact --no-banner --log-opts=--all <repository>
gitleaks dir --redact --no-banner '<repository>/Human Game'
```

The initial history scan reported 41 scanned commits and about 128.72 MB of
content; the cloned repository had 86 reachable commits, including merges.
This is a pattern-based secrets scan of locally fetched history and source, not
a guarantee about personal data, ownership, every ref on GitHub, binary assets,
or the organizers' independent scan. No history was rewritten.

## Remaining limits

- No Firefox, Safari, gamepad, or actual phone/tablet verification yet.
- No full nonvisual play mode or completed accessibility conformance audit.
- Local records are editable by the browser owner; there is no trusted leaderboard.
- Existing ML code and third-party assets elsewhere in the repository were not
  relicensed or certified by creating this component.
- No deployment, external user base, organizer clearance, or onboarding is claimed.
- At v0.1.0, campaign, level editor, replays, and offline installation remained on
  the roadmap. Campaign was delivered in v0.2.0; the updated roadmap excludes it.

## Submission correction verification — 2026-10-17

- Removed the older Snake project's bundled Arial file from the current tree.
  The renderer selects `pygame.font.Font(None, 25)` instead of a repository font.
- With an isolated Python 3.12.13 / Pygame 2.6.1 environment and dummy SDL drivers,
  imported that renderer, rendered score text, created a 640×480 game, and executed
  one non-terminal frame successfully. No model training or inference was run.
- Legacy Python syntax check passed. No `.ttf` file remains in the working tree.
- Switchback `npm run check` and all 29 Node tests passed again after the changes.
- Fresh Gitleaks history and entire working-directory scans reported no leaks.
  The pre-correction-commit history scan reported 42 scanned commits and 128.80 MB;
  the working-directory scan covered approximately 250.23 KB. Reports are redacted.
- Inspected all four CSV objects found in fetched Git history. Each row matched
  its digit-recognition schema: 42,000 training rows with labels 0–9 and 784
  integer pixels 0–255; 28,000 test rows with 784 integer pixels; and two 28,000-row
  prediction files with sequential image IDs and labels 0–9. No unexpected columns
  or nonnumeric cells were found. This verifies those schemas, not all provenance
  or every possible privacy/rights question in repository history.
- Historical Arial copies remain. See `THIRD_PARTY_NOTICES.md` at repository root
  and `ELIGIBILITY.md`; current-tree removal does not grant historical rights.
- Owner confirmed entirely personal work and 20+ hours of development regardless
  of acceptance. No organizer scope or history-remediation decision has been received.

## v0.2.0 campaign release verification

- `npm run check` passed. **42 automated tests passed**, including all original
  rules, six authored completion fixtures, stage data integrity, unlock ordering,
  move-based medals, progress retention, v0.1 save migration, and static build safety.
- Every stage was completed through the actual browser interface using synthetic
  input fixtures: First Light 11 moves, Crosscurrent 37, Slalom 57, Double Lock 67,
  Orbit 45, Afterglow 73. Each awarded Gold and saved; reload showed six completions
  and six unlocked stages. These are QA runs, not claims about real users.
- First Light automatically paused at its first two instruction milestones,
  resumed correctly, completed through its mint gate, unlocked Crosscurrent, and
  advanced through the Next stage button. Reload recommended the first unfinished stage.
- Real button clicks enabled and muted synthesized sound with no page errors.
  At 390×844, a touch pointer swipe moved the rendered snake down and pause worked;
  the page had no horizontal overflow. This is browser emulation, not an actual phone test.
- Clearing recent history preserved campaign progress. Cancelling campaign reset
  preserved it; accepting reset cleared completions and relocked later stages.
- Confirmed both native browser reset-dialog choices separately. With browser
  storage access blocked, the game loaded without page errors and clearing history
  correctly reported that only the current visit could be updated.
- At 1366×768 the campaign board fit within the viewport after starting a run.
- Static build tests verified complete relative module imports, included license,
  exclusion of developer/submission files, stale-file removal, and protection of
  unrelated nonempty directories.
- The runtime has no solver, model, external sound files, telemetry, or remote API.
  Scripted completion fixtures live only in tests and are excluded from the build.
- Actual-device, Firefox/Safari, gamepad, and assistive-technology checks remain
  future work. No deployment, established player base, or program clearance is claimed.

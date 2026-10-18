# Third-party assets and dependencies

The root MIT license covers the owner's project code. It does not grant rights
to third-party assets or replace licenses of separately installed dependencies.

## Human Game / Switchback

The current component includes original JavaScript, HTML, CSS, SVG artwork, and
a screenshot of its interface. It bundles no fonts, audio, external images,
datasets, models, or third-party runtime packages. Its CSS font names refer to
fonts already available on the player's device; font files are not distributed.
Node.js is used to serve files and run tests. See `Human Game/LICENSE`.

## Removed Arial font in the older Snake project

On 2026-10-17, the bundled file at
`Game AI/Snake Game Q Learning/snake_game_deep_q_learning/arial.ttf` was removed
from the current tree. The old Snake renderer now uses `pygame.font.Font(None, 25)`,
which selects the default font from the user's separately installed Pygame package.
No replacement font file is copied into this repository.

The removed file's metadata identified Microsoft/Monotype licensing with
restrictions. The MIT notice inside that file applied to specific Hebrew layout
logic, not to the entire font. No redistribution permission for the font is
claimed here.

Historical Git commits still contain that font. This change preserves the
repository's actual history and does not retroactively license historical copies.
Any removal from history or other remediation must be agreed with the program
organizers before claiming this issue is fully resolved for submission.

- [Microsoft font redistribution guidance](https://learn.microsoft.com/en-us/typography/fonts/font-faq)
- [Pygame default font documentation](https://www.pygame.org/docs/ref/font.html)

## Existing machine-learning projects

The other directories use separately installed packages, with their own licenses.
They and historical data/model artifacts remain outside the proposed Switchback
work scope. Their presence is disclosed; they are not made eligible for CC-CLI v2
by the creation of `Human Game`. This notice is not a certification of all
historical third-party asset rights.

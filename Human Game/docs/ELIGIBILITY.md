# CC-CLI v2 eligibility evidence

**Current verdict: repository preparation is complete for organizer review;
full program eligibility has not been confirmed.**

Source: [submission instructions](https://docs.google.com/document/d/1yShjrmd3D1KenTEwEkheFm_6Pj-FU_vsGpq43ECIriI/edit).
Assessment date: 2026-10-17, America/Los_Angeles. This record distinguishes source
checks and owner statements from decisions only the program can make.

| Requirement | Evidence | Status |
| --- | --- | --- |
| Public GitHub repository | The existing `YouKnowJP/Machine-Learning-Project` repository is public; Human Game is on main. | Verified |
| README: purpose, audience, stack, run instructions | Root README links the component README; both explain setup and scope. | Verified |
| Accepted root LICENSE | Root MIT LICENSE is present, with a matching component copy. | Verified for the owner's code |
| Personal ownership; no employer/client/university/company/NDA claim | Owner explicitly confirmed entirely personal work on 2026-10-17. | Owner-confirmed; formal certification remains theirs |
| 20+ hours of future work the owner intends to do anyway | Owner explicitly confirmed 20+ hours regardless of program acceptance. The selected unfinished roadmap estimates 30–42 hours. | Owner-confirmed; recheck remaining scope at submission |
| No models or prohibited ML work in the proposed component | Human Game imports only its own browser modules and Node built-ins; it has no model or runtime package dependencies. Other repository directories contain model work and are disclosed. | Component verified; mixed-repository scope decision pending |
| Real history and attributable commits | Original history retained; the implementation commit is linked to YouKnowJP on GitHub. | Verified for implementation; no backdating or history rewriting |
| Active project with work ahead | Runnable game, regression tests, CI, a screenshot, changelog, and concrete future milestones. This component is newly created. | Evidence present; maturity/fit assessment pending |
| No committed secrets | Gitleaks scans reported no leaks in fetched history and game source; see QA.md. .env patterns are ignored. | No findings in performed scans; program scan still applies |
| No personal/customer/regulated data in the game | Runtime stores only local settings and scores, with no names, accounts, telemetry, uploads, or remote APIs. All four historical CSV objects were validated as numeric digit pixels/labels or sequential image IDs; see QA.md. | Component and identified CSV schemas verified; no blanket certification of all historical assets |
| Right to distribute assets | Human Game bundles original assets and no third-party font. Restricted Arial removed from the current tree; old renderer uses installed Pygame's default font. Historical copies remain disclosed. | Current-tree correction completed; historical handling pending |
| Visible plan | ROADMAP.md separates completed starter work from 30–42 estimated future hours with acceptance criteria. | Verified |
| Correct submission link | Use https://github.com/YouKnowJP/Machine-Learning-Project and describe Human Game in the form. | Draft prepared; no form submitted |

## Decisions still needed from organizers

1. Whether the newly created Human Game component qualifies inside a repository
   that contains excluded ML projects, and whether its current history is a fit.
2. Whether additional handling of historical Arial copies is required. The current
   font file is removed, no right to redistribute it is claimed, and history has
   not been rewritten. See [third-party notice](../../THIRD_PARTY_NOTICES.md).
3. Final clearance after their repository checks, followed by contributor terms,
   owner certification, and setup instructions.

The instructions provide no fixed minimum project age. A new game should be
described as new; extra documentation commits do not create a long history of use.
Tests, a roadmap, and owner confirmation support review but do not replace it.

The [submission notes](SUBMISSION.md) include one combined scope and historical
font question for General Questions. No organizer message was sent by the coding
assistant, and the owner reported that no decision has been received yet.

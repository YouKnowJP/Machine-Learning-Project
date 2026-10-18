# CC-CLI v2 submission preparation

**Status: candidate project, not cleared or onboarded.** A new non-ML directory
does not guarantee eligibility for a repository whose other components train AI
models. Ask the organizers about this exact scope before submitting it as eligible.

Source: [project submission instructions](https://docs.google.com/document/d/1yShjrmd3D1KenTEwEkheFm_6Pj-FU_vsGpq43ECIriI/edit).
Read the current instructions again before submission; this document is planning
material, not an approval or legal/ownership certification.

## Scope clarification to send yourself

> My public MIT-licensed repository contains older machine-learning projects and
> a new, independently runnable human-played Snake game named Switchback in
> `Human Game`. The existing repository history is preserved. The game has no
> model dependencies, trained agent, inference, or model training/evaluation.
>
> I would propose only further gameplay development: authored campaign stages,
> a local level editor, human-input replays, configurable controls, and offline
> installation. I intend to continue this game for at least 20 hours regardless
> of program acceptance; the selected unfinished backlog is estimated at 30–42 hours.
>
> Can that gameplay-only component qualify within this existing ML repository?
> It is a newly created AI-assisted component, so I am also asking whether its
> current development history meets your project-maturity expectations.
>
> I also removed a restricted Arial font from the current tree of the older Snake
> project and changed that renderer to use Pygame's default font. Historical
> commits still contain the old font. Before changing history, I would like your
> guidance on whether additional remediation is necessary for this submission.
>
> Repository: https://github.com/YouKnowJP/Machine-Learning-Project

This text is a draft for the owner. No message or form has been sent by the coding
assistant. Add ownership claims only after confirming they are true.

## Draft form material: use only after scope clarification

**Project name:** Switchback (Human Game component of Machine Learning Project).

**Repository link:** https://github.com/YouKnowJP/Machine-Learning-Project

The form requests the repository main page, not a branch or subdirectory URL.
Explain the `Human Game` location in the description and comments.

**Short description:**

> Switchback is a human-played browser Snake game for short arcade challenges.
> Players switch between two phases to collect matching energy and pass gates;
> a classic mode and UTC-seeded daily route are also included. It is built with
> vanilla JavaScript, Canvas, HTML, and CSS, with local scores and no backend or
> model dependencies. The proposed next work covers campaign levels, a local
> level editor, replays, configurable controls, and offline installation.

**License:** MIT, as shown in the repository's top-level LICENSE.

**Ownership / company / relationship fields:** Answer personally and accurately.
The owner reported on 2026-10-17 that this repository is entirely personal work,
independent of employment, client, university, company, and confidentiality claims.
GitHub account ownership and a license notice do not establish that the work was
independent of school, clients, employers, company ownership, or confidential work.

**Remaining hours:** Select genuine unfinished work from [ROADMAP.md](../ROADMAP.md)
and update the estimates as work progresses. On 2026-10-17, the owner explicitly
confirmed an intent to do 20+ hours regardless of acceptance. Do not count the
completed starter implementation toward the program's future-work requirement.

**Anything else:** Disclose the ML components elsewhere in this repository, link
the roadmap, mention the new AI-assisted game component, and cite any organizer
scope decision. Do not claim existing users, deployments, releases, maintenance
history, clearance, or scan results that have not been verified.

## Before sending the form

- Confirm organizer acceptance of the mixed-repository and new-component scope.
- Obtain organizer guidance on the historical Arial font copies; current-tree
  removal has been completed, and no redistribution permission is claimed.
- Ensure the reviewed game files are present on the repository's main branch.
- Confirm your ownership, resource independence, NDA/company status, and intent
  to complete at least 20 hours of unfinished work.
- Review the full repository and history for secrets and personal/customer data.
  If anything is found, revoke credentials and follow the program's remediation
  instructions; do not rewrite history merely to make the project appear older
  or different. Preserve genuine history except necessary, authorized remediation.
- Keep the root MIT LICENSE, working setup instructions, tests, and visible plans.
- Submit accurate information in Labelbox, then wait for clearance and contributor
  terms/setup instructions. This source change itself does not onboard you.

See [ELIGIBILITY.md](ELIGIBILITY.md) for the requirement-by-requirement record.

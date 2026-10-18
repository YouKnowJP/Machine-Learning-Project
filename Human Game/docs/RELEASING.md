# Release instructions

Run these from `Human Game` with Node.js 22+:

```sh
npm run check
npm test
npm run build
```

The build produces `dist/`, containing the playable page, styles, icon, five browser
modules, MIT license, release instructions, and a small build manifest. It copies
from an explicit public-file list. Legacy ML directories, repository history,
tests, server code, and submission documents are not included in the static asset.
The source remains in the original GitHub repository with its history intact.

Serve `dist/` over HTTP to smoke-test the package:

```sh
python3 -m http.server 4174 --bind 127.0.0.1 --directory dist
```

Check first-run lessons, stage unlocks, saved medals, arcade modes, pause/resume,
touch controls, sound mute, and keyboard navigation. Verify that an existing v0.1
save is retained and that blocked storage still permits play. Broader actual-device
and assistive-technology verification remains on the roadmap.

## Publishing a release

1. Update `package.json`, the displayed version in `index.html`, and the changelog.
2. Complete checks and build from the same source revision. Inspect the asset list.
3. Package only the built directory; keep the LICENSE and RELEASE.md with it.
4. Attach that package to a release if desired, linking its actual source commit.
5. State the tested browsers and remaining limits accurately. Do not claim that
   shipping a version constitutes CC-CLI clearance or establishes historical usage.

No third-party hosting or GitHub release is automatically published by the build.
The development server binds to loopback. A static host needs no server, secrets,
accounts, external fonts, model services, or installation step for the player.

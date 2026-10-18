# Release instructions

Run these from `Human Game` with Node.js 22+:

```sh
npm run check
npm test
npm run build
```

The build produces `dist/`, containing the playable page, styles, icon, five browser
modules, MIT license, release instructions, and a small build manifest. It copies
from an explicit public-file list. Other repository projects, tests, server code,
and development documents are excluded.

Serve `dist/` over HTTP to smoke-test the package:

```sh
python3 -m http.server 4174 --bind 127.0.0.1 --directory dist
```

Check first-run lessons, stage unlocks, saved medals, arcade modes, pause/resume,
touch controls, sound mute, and keyboard navigation. Verify that an existing v0.1
save is retained and that blocked storage still permits play. Record the tested
browsers and devices in [QA.md](QA.md).

## Publish a release

1. Update `package.json`, the displayed version in `index.html`, and the changelog.
2. Run checks, tests, and the build from the same source revision.
3. Inspect the asset list and quick-start instructions in `dist/RELEASE.md`.
4. Archive only the contents of `dist/`, keeping `LICENSE` and `RELEASE.md`.
5. Generate a SHA-256 checksum for the archive.
6. Tag the tested commit as `switchback-v<version>`, matching the versioned source
   link generated in `RELEASE.md`. Attach the archive and checksum to that release.
7. Include player-facing changes, launch instructions, and verification limits
   in the release notes. Download the published archive and verify its checksum.

The build creates local files; publishing is a separate step. The development
server binds to loopback. Players can use the static package with any HTTP server.

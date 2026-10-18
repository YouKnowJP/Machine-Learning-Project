import { cp, mkdir, mkdtemp, readFile, readdir, rename, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = fileURLToPath(new URL('../', import.meta.url));
export const PUBLIC_FILES = Object.freeze([
  'index.html', 'style.css', 'icon.svg', 'LICENSE',
  'src/app.mjs', 'src/engine.mjs', 'src/storage.mjs', 'src/levels.mjs', 'src/audio.mjs',
]);

export async function build(destination = path.join(ROOT, 'dist')) {
  destination = path.resolve(destination);
  const marker = '.switchback-build.json';
  let existing;
  try { existing = await readdir(destination); }
  catch (error) { if (error.code !== 'ENOENT') throw error; }
  if (existing?.length) {
    let owned = false;
    try { owned = JSON.parse(await readFile(path.join(destination, marker), 'utf8')).project === 'switchback'; }
    catch { /* An unknown directory must not be overwritten. */ }
    if (!owned) throw new Error('Build destination is not an existing Switchback build; choose an empty directory.');
  }
  const parent = path.dirname(destination);
  await mkdir(parent, { recursive: true });
  const staging = await mkdtemp(path.join(parent, '.switchback-build-'));
  try {
    for (const file of PUBLIC_FILES) {
      await mkdir(path.dirname(path.join(staging, file)), { recursive: true });
      await cp(path.join(ROOT, file), path.join(staging, file));
    }
    const { version } = JSON.parse(await readFile(path.join(ROOT, 'package.json'), 'utf8'));
    await writeFile(path.join(staging, marker), JSON.stringify({ project: 'switchback', version, files: PUBLIC_FILES }, null, 2) + '\n');
    await writeFile(path.join(staging, 'RELEASE.md'), `# Switchback ${version}\n\nHuman-played Snake with six campaign stages, arcade modes, local progress, and optional synthesized sound.\n\nServe this directory over HTTP. For example, run:\n\n\`\`\`sh\npython3 -m http.server 4173 --bind 127.0.0.1\n\`\`\`\n\nThen open http://127.0.0.1:4173. Do not open index.html through file://, because ES modules need an HTTP origin.\n\nMIT; see LICENSE. No accounts, models, third-party runtime packages, or external asset downloads.\n\nSource and original history: https://github.com/YouKnowJP/Machine-Learning-Project/tree/main/Human%20Game\n\nThis static game package is not evidence of CC-CLI v2 approval.\n`);
    if (existing) await rm(destination, { recursive: true });
    await rename(staging, destination);
    return { version, destination, files: [...PUBLIC_FILES, marker, 'RELEASE.md'] };
  } catch (error) {
    await rm(staging, { recursive: true, force: true });
    throw error;
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const result = await build();
  console.log(`Switchback ${result.version}: ${result.files.length} static files ready in ${result.destination}`);
}

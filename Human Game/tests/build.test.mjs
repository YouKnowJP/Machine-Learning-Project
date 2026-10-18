import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { build, PUBLIC_FILES } from '../scripts/build.mjs';

test('static release includes all modules and the license, and preserves no stale files', async () => {
  const temporary = await mkdtemp(path.join(tmpdir(), 'switchback-test-'));
  const destination = path.join(temporary, 'release');
  try {
    const result = await build(destination);
    assert.equal(result.version, '0.2.0');
    for (const file of PUBLIC_FILES) assert.ok((await readFile(path.join(destination, file))).length);
    for (const file of PUBLIC_FILES.filter(file => file.endsWith('.mjs'))) {
      const source = await readFile(path.join(destination, file), 'utf8');
      for (const [, relative] of source.matchAll(/from\s+['"]([^'"]+)['"]/g)) {
        assert.ok(relative.startsWith('./'), 'Browser modules must be contained in the release');
        assert.ok((await readFile(path.resolve(destination, path.dirname(file), relative))).length);
      }
    }
    await assert.rejects(readFile(path.join(destination, 'serve.mjs')));
    await assert.rejects(readFile(path.join(destination, 'docs/SUBMISSION.md')));
    await writeFile(path.join(destination, 'stale-file.txt'), 'stale');
    await build(destination);
    assert.ok(!(await readdir(destination)).includes('stale-file.txt'));
  } finally { await rm(temporary, { recursive: true, force: true }); }
});

test('build refuses to overwrite an unrelated nonempty directory', async () => {
  const temporary = await mkdtemp(path.join(tmpdir(), 'switchback-test-'));
  try {
    await writeFile(path.join(temporary, 'keep.txt'), 'keep');
    await assert.rejects(build(temporary), /not an existing Switchback build/);
    assert.equal(await readFile(path.join(temporary, 'keep.txt'), 'utf8'), 'keep');
  } finally { await rm(temporary, { recursive: true, force: true }); }
});

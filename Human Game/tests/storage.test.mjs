import { test } from 'node:test';
import assert from 'node:assert/strict';
import { loadProfile, saveProfile, clearProfile, emptyProfile, recordRun } from '../src/storage.mjs';
import { createGame, scoreKey } from '../src/engine.mjs';

const memory = () => { const values = new Map(); return { getItem: key => values.get(key) ?? null, setItem: (key, value) => values.set(key, value), removeItem: key => values.delete(key) }; };
test('empty storage loads a new profile', () => assert.deepEqual(loadProfile(memory()), emptyProfile()));
test('preferences and scores round trip', () => {
  const storage = memory(), profile = emptyProfile(); profile.difficulty = 'rush'; profile.contrast = true; profile.bests['phase:rush'] = 300;
  assert.equal(saveProfile(storage, profile), true); assert.deepEqual(loadProfile(storage), profile);
});
test('blocked storage does not break loading, saving, or clearing', () => {
  const blocked = { getItem() { throw new Error('blocked'); }, setItem() { throw new Error('quota'); }, removeItem() { throw new Error('blocked'); } };
  assert.deepEqual(loadProfile(blocked), emptyProfile()); assert.equal(saveProfile(blocked, emptyProfile()), false); assert.equal(clearProfile(blocked), false);
});
test('corrupt JSON, invalid settings, and invalid records are discarded', () => {
  assert.deepEqual(loadProfile({ getItem: () => '{broken' }), emptyProfile());
  const loaded = loadProfile({ getItem: () => JSON.stringify({ bests: { 'phase:standard': 100, 'classic:standard': -2, bad: 7 }, difficulty: 'fastest', runs: [null, { mode: 'unknown', score: 1 }], contrast: 'yes' }) });
  assert.deepEqual(loaded.bests, { 'phase:standard': 100 }); assert.equal(loaded.difficulty, 'standard'); assert.deepEqual(loaded.runs, []); assert.equal(loaded.contrast, false);
});
test('recording keeps five recent runs and never lowers a personal best', () => {
  const profile = emptyProfile(), game = createGame(); game.score = 500; recordRun(profile, game);
  game.score = 100; for (let i = 0; i < 6; i++) recordRun(profile, game);
  assert.equal(profile.bests[scoreKey(game)], 500); assert.equal(profile.runs.length, 5);
});
test('clearing removes persisted records', () => {
  const storage = memory(); saveProfile(storage, { ...emptyProfile(), bests: { 'phase:standard': 100 } });
  assert.equal(clearProfile(storage), true); assert.deepEqual(loadProfile(storage), emptyProfile());
});

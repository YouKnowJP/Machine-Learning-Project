import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { LEVELS, availableLevelIds, recommendedLevelId, medalFor } from '../src/levels.mjs';
import { createGame, startGame, queueDirection, shiftPhase, step, scoreKey, SIZE, tickDuration } from '../src/engine.mjs';

test('authored stages have valid, unique cells and unobstructed objectives', () => {
  assert.equal(LEVELS.length, 6);
  assert.equal(new Set(LEVELS.map(level => level.id)).size, 6);
  for (const level of LEVELS) {
    const arena = [...level.walls, ...level.gates], blocked = new Set();
    for (const point of arena) {
      assert.ok(point.x >= 0 && point.x < SIZE && point.y >= 0 && point.y < SIZE, level.id);
      assert.ok(!blocked.has(`${point.x},${point.y}`), `${level.id}: duplicate arena cell`);
      blocked.add(`${point.x},${point.y}`);
    }
    const game = createGame({ mode: 'campaign', levelId: level.id });
    assert.ok(game.snake.every(point => !blocked.has(`${point.x},${point.y}`)), level.id);
    for (const point of level.foods) {
      assert.ok(point.x >= 0 && point.x < SIZE && point.y >= 0 && point.y < SIZE);
      assert.ok(!blocked.has(`${point.x},${point.y}`), `${level.id}: blocked energy`);
      assert.ok([0, 1].includes(point.phase));
    }
  }
});

test('First Light teaches a complete three-pickup route', () => {
  const game = startGame(createGame({ mode: 'campaign' }));
  assert.equal(tickDuration(game), 220);
  for (let i = 0; i < 3; i++) step(game);
  assert.equal(game.collected, 1);
  shiftPhase(game); queueDirection(game, 'up');
  for (let i = 0; i < 3; i++) step(game);
  assert.equal(game.collected, 2);
  shiftPhase(game); queueDirection(game, 'right');
  for (let i = 0; i < 5; i++) step(game);
  assert.equal(game.status, 'won'); assert.equal(game.score, 300); assert.equal(game.ticks, 11);
  assert.equal(medalFor(game), 3); assert.equal(game.food, null);
});

test('campaign layout data is not mutated by a live game', () => {
  const before = JSON.stringify(LEVELS);
  const game = createGame({ mode: 'campaign' }); game.gates[0].phase = 1; game.food.x = 2;
  assert.equal(JSON.stringify(LEVELS), before);
});

test('stages unlock in order and the next unfinished stage is recommended', () => {
  assert.deepEqual(availableLevelIds({}), ['first-light']);
  const progress = { 'first-light': { medal: 3, ticks: 11 } };
  assert.deepEqual(availableLevelIds(progress), ['first-light', 'crosscurrent']);
  assert.equal(recommendedLevelId(progress), 'crosscurrent');
  const all = Object.fromEntries(LEVELS.map(level => [level.id, { medal: 1, ticks: 100 }]));
  assert.equal(availableLevelIds(all).length, 6); assert.equal(recommendedLevelId(all), 'afterglow');
});

test('medals use move count and are never awarded for an unfinished run', () => {
  const game = createGame({ mode: 'campaign' });
  assert.equal(medalFor(game), 0);
  game.status = 'won'; game.ticks = 14; assert.equal(medalFor(game), 3);
  game.ticks = 21; assert.equal(medalFor(game), 2);
  game.ticks = 22; assert.equal(medalFor(game), 1);
});

test('campaign records are keyed by stage and pace', () => {
  assert.notEqual(scoreKey(createGame({ mode: 'campaign' })), scoreKey(createGame({ mode: 'campaign', levelId: 'orbit' })));
  assert.throws(() => createGame({ mode: 'campaign', levelId: 'unknown' }));
});

test('every authored stage has an executable collision-free completion fixture', async () => {
  const routes = JSON.parse(await readFile(new URL('./fixtures/campaign-routes.json', import.meta.url), 'utf8'));
  assert.equal(routes.length, LEVELS.length);
  for (const route of routes) {
    const game = startGame(createGame({ mode: 'campaign', levelId: route.levelId }));
    for (const move of route.moves) {
      if (game.phase !== move.phase) shiftPhase(game);
      if (game.direction !== move.direction) assert.equal(queueDirection(game, move.direction), true);
      const result = step(game);
      assert.notEqual(result.type, 'crash', `${route.levelId}: ${game.reason}`);
    }
    assert.equal(game.status, 'won', route.levelId);
    assert.equal(game.collected, game.target);
  }
});

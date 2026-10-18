import { test } from 'node:test';
import assert from 'node:assert/strict';
import { SIZE, createGame, startGame, queueDirection, shiftPhase, togglePause, step, tickDuration, dailySeed, scoreKey, same } from '../src/engine.mjs';

const running = (options = {}) => startGame(createGame(options));
function foodAhead(game, phase = game.phase) {
  game.food = { x: game.snake[0].x + 1, y: game.snake[0].y, phase };
}

test('a run waits for explicit start and never moves while ready', () => {
  const game = createGame(), before = structuredClone(game);
  assert.equal(step(game).type, 'idle'); assert.deepEqual(game, before);
});
test('movement follows one grid cell per tick without growing', () => {
  const game = running(); game.food = { x: 20, y: 20, phase: 0 };
  assert.equal(step(game).type, 'move'); assert.equal(game.snake[0].x, 6); assert.equal(game.snake.length, 4);
});
test('an opposite turn is rejected', () => {
  const game = running(); assert.equal(queueDirection(game, 'left'), false); step(game); assert.equal(game.direction, 'right');
});
test('rapid turns cannot reverse the snake within a single simulation tick', () => {
  const game = running(); assert.equal(queueDirection(game, 'up'), true); assert.equal(queueDirection(game, 'left'), false);
  step(game); assert.equal(game.direction, 'up'); assert.equal(queueDirection(game, 'left'), true);
});
test('same-direction input does not consume the turn buffer', () => {
  const game = running(); assert.equal(queueDirection(game, 'right'), false); assert.equal(queueDirection(game, 'down'), true);
});
test('invalid directions, modes, difficulty, and seeds are rejected', () => {
  assert.equal(queueDirection(running(), '__proto__'), false);
  assert.throws(() => createGame({ mode: 'training' }));
  assert.throws(() => createGame({ difficulty: 'unknown' }));
  assert.throws(() => createGame({ seed: '' }));
});
test('matching energy increases score by 100 and grows by one', () => {
  const game = running(); foodAhead(game);
  assert.equal(step(game).type, 'collect'); assert.equal(game.score, 100); assert.equal(game.snake.length, 5);
  assert.equal(game.food.phase, 1);
});
test('unmatched energy stays in place without growing the snake', () => {
  const game = running(); foodAhead(game, 1); const food = { ...game.food };
  assert.equal(step(game).type, 'move'); assert.equal(game.score, 0); assert.equal(game.snake.length, 4); assert.deepEqual(game.food, food);
});
test('classic has no obstacles or gates and does not need phase matching', () => {
  const game = running({ mode: 'classic' }); foodAhead(game, 1);
  assert.equal(shiftPhase(game), false); assert.equal(game.gates.length + game.walls.length, 0);
  assert.equal(step(game).type, 'collect'); assert.equal(game.food.phase, 0);
});
test('a matching phase passes a gate', () => {
  const game = running(); game.gates = [{ x: 6, y: 12, phase: 0 }]; game.food = { x: 20, y: 20, phase: 0 };
  assert.equal(step(game).type, 'move'); assert.equal(game.snake[0].x, 6);
});
test('a wrong-phase gate ends the run', () => {
  const game = running(); game.gates = [{ x: 6, y: 12, phase: 1 }];
  assert.equal(step(game).type, 'crash'); assert.match(game.reason, /Wrong phase/); assert.equal(game.status, 'over');
});
test('shifting allows the next gate entry and records the head phase', () => {
  const game = running(); game.gates = [{ x: 6, y: 12, phase: 1 }];
  assert.equal(shiftPhase(game), true); step(game); assert.equal(game.status, 'running'); assert.equal(game.snake[0].phase, 1);
});
test('solid blocks and board edges cause a crash', () => {
  const wall = running(); wall.walls = [{ x: 6, y: 12 }]; assert.equal(step(wall).type, 'crash');
  const edge = running(); edge.snake = [{ x: SIZE - 1, y: 12, phase: 0 }]; assert.equal(step(edge).type, 'crash');
});
test('body collision ends a run', () => {
  const game = running(); game.snake = [{ x: 5, y: 12 }, { x: 6, y: 12 }, { x: 6, y: 13 }];
  assert.equal(step(game).type, 'crash'); assert.match(game.reason, /own tail/);
});
test('entering a tail cell that vacates on this step is legal', () => {
  const game = running(); game.snake = [{ x: 5, y: 12 }, { x: 5, y: 13 }, { x: 6, y: 13 }, { x: 6, y: 12 }];
  game.food = { x: 20, y: 20, phase: 0 }; assert.equal(step(game).type, 'move'); assert.equal(game.status, 'running');
});
test('pause freezes simulation and rejects game input', () => {
  const game = running(); togglePause(game); const before = structuredClone(game);
  assert.equal(shiftPhase(game), false); assert.equal(queueDirection(game, 'up'), false); assert.equal(step(game).type, 'idle');
  assert.deepEqual(game, before); togglePause(game); assert.equal(game.status, 'running');
});
test('a full board wins instead of looping forever while spawning food', () => {
  const game = running({ mode: 'classic' });
  game.snake = [{ x: 0, y: 0, phase: 0 }];
  for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) {
    if (!(x === 0 && y === 0) && !(x === 1 && y === 0)) game.snake.push({ x, y, phase: 0 });
  }
  game.food = { x: 1, y: 0, phase: 0 };
  assert.equal(step(game).type, 'win'); assert.equal(game.status, 'won'); assert.equal(game.food, null); assert.equal(game.snake.length, SIZE * SIZE);
});
test('same seed and input sequence produce identical runs', () => {
  const a = running({ seed: 'repeatable-route' }), b = running({ seed: 'repeatable-route' });
  for (let tick = 0; tick < 20; tick++) {
    if ([3, 8, 13, 18].includes(tick)) { const direction = { 3: 'down', 8: 'left', 13: 'up', 18: 'right' }[tick]; queueDirection(a, direction); queueDirection(b, direction); }
    if (tick % 4 === 0) { shiftPhase(a); shiftPhase(b); }
    step(a); step(b);
  }
  assert.deepEqual(a, b);
});
test('food avoids snake, gates, and walls across many route seeds', () => {
  for (let index = 0; index < 500; index++) {
    const game = createGame({ seed: `route-${index}` });
    assert.ok(![...game.snake, ...game.gates, ...game.walls].some(point => same(point, game.food)));
  }
});
test('daily date uses UTC and forces standard pace', () => {
  assert.equal(dailySeed(new Date('2026-10-17T20:00:00-07:00')), 'daily-2026-10-18');
  const game = createGame({ mode: 'daily', difficulty: 'rush', seed: 'daily-2026-10-18' });
  assert.equal(game.difficulty, 'standard'); assert.equal(scoreKey(game), 'daily:standard:daily-2026-10-18');
});
test('records are separated by mode and difficulty', () => {
  assert.notEqual(scoreKey(createGame()), scoreKey(createGame({ mode: 'classic' })));
  assert.notEqual(scoreKey(createGame()), scoreKey(createGame({ difficulty: 'rush' })));
});
test('pace ramps gradually and never drops below 65 milliseconds', () => {
  const game = createGame({ difficulty: 'rush' }); assert.equal(tickDuration(game), 95);
  game.collected = 3; assert.equal(tickDuration(game), 90); game.collected = 999; assert.equal(tickDuration(game), 65);
});
test('completed runs are stable and reject movement or phase changes', () => {
  const game = running(); game.walls = [{ x: 6, y: 12 }]; step(game); const before = structuredClone(game);
  assert.equal(shiftPhase(game), false); assert.equal(queueDirection(game, 'up'), false); step(game); assert.deepEqual(game, before);
});

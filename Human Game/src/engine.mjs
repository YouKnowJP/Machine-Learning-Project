// Pure game rules. No DOM, clock, storage, network, or learned policy.
export const SIZE = 24;
export const VERSION = 1;
export const DIRECTIONS = Object.freeze({
  up: { x: 0, y: -1 }, right: { x: 1, y: 0 },
  down: { x: 0, y: 1 }, left: { x: -1, y: 0 },
});
export const SPEEDS = Object.freeze({ cruise: 170, standard: 130, rush: 95 });
export const same = (a, b) => a.x === b.x && a.y === b.y;
export const key = (point) => `${point.x},${point.y}`;

export function hashSeed(text) {
  let hash = 2166136261;
  for (const char of String(text)) hash = Math.imul(hash ^ char.charCodeAt(0), 16777619);
  return hash >>> 0;
}

// xorshift32; the state is stored on the game, so runs are reproducible.
function random(game) {
  let value = game.rng || 1;
  value ^= value << 13; value ^= value >>> 17; value ^= value << 5;
  game.rng = value >>> 0;
  return game.rng / 4294967296;
}

export function dailySeed(date = new Date()) {
  return `daily-${date.toISOString().slice(0, 10)}`;
}

export function makeArena(mode, seed) {
  const walls = [], gates = [];
  if (mode === 'classic') return { walls, gates };
  // Two broken circuit dividers. Gaps always leave routes between sections.
  for (const x of [8, 15]) {
    for (const y of [3, 4, 5, 6, 7, 16, 17, 18, 19, 20]) {
      const phase = ((x === 8 ? 0 : 1) + (y > 12 ? 1 : 0) + (hashSeed(seed) % 2)) % 2;
      if (y === 3 || y === 20) walls.push({ x, y });
      else gates.push({ x, y, phase });
    }
  }
  for (const x of [3, 4, 5, 18, 19, 20]) {
    walls.push({ x, y: 3 }, { x, y: 20 });
  }
  return { walls, gates };
}

function spawnFood(game) {
  const blocked = new Set([...game.snake, ...game.walls, ...game.gates].map(key));
  const empty = [];
  for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) {
    if (!blocked.has(`${x},${y}`)) empty.push({ x, y });
  }
  if (!empty.length) {
    game.food = null;
    game.status = 'won';
    game.reason = 'Circuit complete. Every energy cell is filled.';
    return;
  }
  const position = empty[Math.floor(random(game) * empty.length)];
  game.food = { ...position, phase: game.mode === 'classic' ? 0 : game.collected % 2 };
}

export function createGame({ mode = 'phase', difficulty = 'standard', seed = 'switchback' } = {}) {
  if (!['phase', 'classic', 'daily'].includes(mode)) throw new Error('Unknown game mode');
  if (!Object.hasOwn(SPEEDS, difficulty)) throw new Error('Unknown difficulty');
  if (typeof seed !== 'string' || seed.length > 80 || !seed.trim()) throw new Error('Invalid route seed');
  const game = {
    version: VERSION, mode, difficulty: mode === 'daily' ? 'standard' : difficulty,
    seed, rng: hashSeed(seed) || 1, status: 'ready', reason: '',
    snake: [5, 4, 3, 2].map(x => ({ x, y: 12, phase: 0 })),
    direction: 'right', queued: null, phase: 0, score: 0, collected: 0,
    ticks: 0, switches: 0, food: null, ...makeArena(mode, seed),
  };
  spawnFood(game);
  return game;
}

export function startGame(game) {
  if (game.status === 'ready') game.status = 'running';
  return game;
}

export function queueDirection(game, direction) {
  if (game.status !== 'running' || !Object.hasOwn(DIRECTIONS, direction) || game.queued) return false;
  const current = DIRECTIONS[game.direction], next = DIRECTIONS[direction];
  if (current.x + next.x === 0 && current.y + next.y === 0) return false;
  if (direction === game.direction) return false;
  game.queued = direction;
  return true;
}

export function shiftPhase(game) {
  if (game.status !== 'running' || game.mode === 'classic') return false;
  game.phase = 1 - game.phase;
  game.switches++;
  return true;
}

export function togglePause(game) {
  if (game.status === 'running') game.status = 'paused';
  else if (game.status === 'paused') game.status = 'running';
  return game;
}

export function step(game) {
  if (game.status !== 'running') return { type: 'idle' };
  if (game.queued) game.direction = game.queued;
  game.queued = null;
  const delta = DIRECTIONS[game.direction];
  const head = { x: game.snake[0].x + delta.x, y: game.snake[0].y + delta.y, phase: game.phase };
  const eats = game.food && same(head, game.food) && (game.mode === 'classic' || game.food.phase === game.phase);
  // The tail vacates its square on a non-growing step.
  const body = eats ? game.snake : game.snake.slice(0, -1);
  const gate = game.gates.find(cell => same(head, cell));
  let reason = '';
  if (head.x < 0 || head.x >= SIZE || head.y < 0 || head.y >= SIZE) reason = 'You reached the edge of the circuit.';
  else if (game.walls.some(cell => same(head, cell))) reason = 'A solid block stopped your run.';
  else if (body.some(cell => same(head, cell))) reason = 'Your route crossed its own tail.';
  else if (gate && gate.phase !== game.phase) reason = 'Wrong phase. Shift before entering a gate.';
  game.ticks++;
  if (reason) {
    game.status = 'over'; game.reason = reason;
    return { type: 'crash', reason };
  }
  game.snake.unshift(head);
  if (eats) {
    game.collected++;
    game.score += 100;
    spawnFood(game);
    return { type: game.status === 'won' ? 'win' : 'collect' };
  }
  game.snake.pop();
  return { type: 'move' };
}

export function tickDuration(game) {
  // Readable early game, then a gentle ramp with a lower bound.
  return Math.max(65, SPEEDS[game.difficulty] - Math.floor(game.collected / 3) * 5);
}

export function scoreKey(game) {
  return `${game.mode}:${game.difficulty}${game.mode === 'daily' ? `:${game.seed}` : ''}`;
}

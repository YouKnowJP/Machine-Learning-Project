// Authored human-played stages. No solver or automated player is shipped.
const point = (x, y) => ({ x, y });
const food = (x, y, phase) => ({ x, y, phase });

function divider(x, openings) {
  const walls = [], gates = [];
  for (let y = 3; y <= 20; y++) {
    if (Object.hasOwn(openings, y)) gates.push({ x, y, phase: openings[y] });
    else walls.push(point(x, y));
  }
  return { walls, gates };
}

function slalom() {
  const walls = [], gates = [];
  for (const [y, openings] of [[6, { 6: 0, 17: 1 }], [11, { 9: 1, 18: 0 }], [17, { 5: 0, 14: 1 }]]) {
    for (let x = 4; x <= 19; x++) {
      if (Object.hasOwn(openings, x)) gates.push({ x, y, phase: openings[x] });
      else walls.push(point(x, y));
    }
  }
  return { walls, gates };
}

function orbit() {
  const walls = [], gates = [];
  for (const [left, right, top, bottom] of [[8, 20, 4, 20], [12, 16, 8, 16]]) {
    const openings = new Map([
      [`${left},12`, left === 8 ? 0 : 1], [`${right},12`, right === 20 ? 1 : 0],
      [`14,${top}`, top === 4 ? 1 : 0], [`14,${bottom}`, bottom === 20 ? 0 : 1],
    ]);
    for (let y = top; y <= bottom; y++) for (let x = left; x <= right; x++) {
      if (x !== left && x !== right && y !== top && y !== bottom) continue;
      const phase = openings.get(`${x},${y}`);
      if (phase !== undefined) gates.push({ x, y, phase });
      else walls.push(point(x, y));
    }
  }
  return { walls, gates };
}

function afterglow() {
  const walls = [], gates = [];
  for (const value of [...Array.from({ length: 8 }, (_, i) => i + 2), ...Array.from({ length: 8 }, (_, i) => i + 14)]) {
    if ([5, 18].includes(value)) {
      gates.push({ x: 11, y: value, phase: value === 5 ? 0 : 1 });
      gates.push({ x: value, y: 11, phase: value === 5 ? 1 : 0 });
    } else walls.push(point(11, value), point(value, 11));
  }
  return { walls, gates };
}

const crossed = divider(11, { 6: 0, 12: 1, 18: 0 });
const lockA = divider(9, { 6: 0, 12: 1, 18: 0 });
const lockB = divider(15, { 6: 1, 12: 0, 18: 1 });

const stages = [
  {
    id: 'first-light', name: 'First Light', label: 'LEARN THE RHYTHM', par: 14,
    description: 'Three energy cells. One new instinct. Learn movement, shifting, and your first gate.',
    walls: [], gates: Array.from({ length: 9 }, (_, i) => ({ x: 10, y: i + 8, phase: 0 })),
    foods: [food(8, 12, 0), food(8, 9, 1), food(13, 9, 0)],
    hints: [
      'Move right to the mint circle. Arrow keys or WASD steer your snake.',
      'Press Space to become amber, then turn up toward the diamond.',
      'Switch back to mint. Turn right and pass through the matching gate.',
    ],
  },
  {
    id: 'crosscurrent', name: 'Crosscurrent', label: 'READ THE OPENINGS', par: 48,
    description: 'A divider changes the route. Find an opening, match its phase, and keep a way out.',
    ...crossed, foods: [food(8, 8, 0), food(15, 8, 1), food(15, 16, 0), food(8, 16, 1)],
  },
  {
    id: 'slalom', name: 'Slalom', label: 'THREAD THE LANES', par: 72,
    description: 'Weave through three offset lanes. The shortest line is not always the clearest one.',
    ...slalom(), foods: [food(7, 14, 0), food(7, 8, 1), food(16, 8, 0), food(16, 4, 1), food(16, 20, 0)],
  },
  {
    id: 'double-lock', name: 'Double Lock', label: 'TWO GATES. TWO PHASES.', par: 86,
    description: 'The dividers disagree. Change phase between gates and plan your next exit.',
    walls: [...lockA.walls, ...lockB.walls], gates: [...lockA.gates, ...lockB.gates],
    foods: [food(7, 5, 0), food(18, 5, 1), food(18, 19, 0), food(12, 15, 1), food(6, 19, 0), food(6, 8, 1)],
  },
  {
    id: 'orbit', name: 'Orbit', label: 'INSIDE THE CIRCUIT', par: 60,
    description: 'Enter two nested rings, then find your way back out before your tail closes the route.',
    ...orbit(), foods: [food(10, 12, 0), food(14, 12, 1), food(18, 12, 0), food(18, 6, 1), food(10, 18, 0), food(14, 18, 1)],
  },
  {
    id: 'afterglow', name: 'Afterglow', label: 'MAKE IT ALL CONNECT', par: 94,
    description: 'The final crossing. Collect seven cells and bring the whole circuit to life.',
    ...afterglow(), foods: [food(5, 16, 0), food(16, 16, 1), food(16, 5, 0), food(5, 5, 1), food(8, 8, 0), food(15, 15, 1), food(19, 19, 0)],
  },
];

function freeze(value) {
  if (value && typeof value === 'object') {
    Object.values(value).forEach(freeze); Object.freeze(value);
  }
  return value;
}
export const LEVELS = freeze(stages);
export function getLevel(id) {
  const level = LEVELS.find(candidate => candidate.id === id);
  if (!level) throw new Error('Unknown campaign stage');
  return level;
}
export function availableLevelIds(progress = {}) {
  const available = [];
  for (const level of LEVELS) {
    available.push(level.id);
    if (!progress[level.id]) break;
  }
  return available;
}
export function recommendedLevelId(progress = {}) {
  return LEVELS.find(level => !progress[level.id])?.id || LEVELS.at(-1).id;
}
export function medalFor(game) {
  if (game.mode !== 'campaign' || game.status !== 'won') return 0;
  const { par } = getLevel(game.levelId);
  return game.ticks <= par ? 3 : game.ticks <= Math.ceil(par * 1.5) ? 2 : 1;
}
export const MEDAL_NAMES = Object.freeze(['Not completed', 'Bronze', 'Silver', 'Gold']);

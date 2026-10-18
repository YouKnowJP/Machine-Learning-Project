import { SIZE, createGame, startGame, queueDirection, shiftPhase, togglePause, step, tickDuration, scoreKey, dailySeed } from './engine.mjs';
import { loadProfile, saveProfile, clearProfile, recordRun, emptyProfile } from './storage.mjs';

const $ = id => document.getElementById(id);
const canvas = $('board'), ctx = canvas.getContext('2d');
const width = 672, cell = width / SIZE;
const ratio = Math.min(window.devicePixelRatio || 1, 2);
canvas.width = width * ratio; canvas.height = width * ratio;
ctx.scale(ratio, ratio);
let storage;
try { storage = window.localStorage; }
catch { storage = { getItem: () => null, setItem: () => { throw new Error('Storage unavailable'); }, removeItem: () => { throw new Error('Storage unavailable'); } }; }
let profile = loadProfile(storage), mode = 'phase';
let game, accumulator = 0, previousFrame = 0, flashUntil = 0, toastTimeout;
const modeNames = { phase: 'Phase run', classic: 'Classic', daily: 'Daily circuit' };
const colors = () => document.body.classList.contains('high-contrast') ? ['#c1ff8b', '#ffd293'] : ['#b3f876', '#ffc078'];
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
$('difficulty').value = profile.difficulty;
document.body.classList.toggle('high-contrast', profile.contrast);

function announce(message) { $('announcer').textContent = message; }
function toast(message) {
  $('toast').textContent = message; $('toast').classList.add('visible');
  clearTimeout(toastTimeout);
  toastTimeout = setTimeout(() => $('toast').classList.remove('visible'), 3000);
}
function persist() {
  if (!saveProfile(storage, profile)) $('storage-note').textContent = 'Storage unavailable. Scores last for this visit only.';
}
function renderRecords() {
  const list = $('runs'); list.replaceChildren();
  if (!profile.runs.length) {
    const item = document.createElement('li'); item.className = 'empty-runs';
    item.textContent = 'A clean slate. Your first circuit starts here.'; list.append(item);
  }
  profile.runs.forEach((run, index) => {
    const item = document.createElement('li'), label = document.createElement('span');
    const number = document.createElement('span'); number.className = 'run-index';
    number.textContent = String(index + 1).padStart(2, '0');
    label.append(number, modeNames[run.mode]);
    const score = document.createElement('strong'); score.textContent = String(run.score).padStart(4, '0');
    item.append(label, score); list.append(item);
  });
}

function newGame() {
  if (mode === 'daily') $('seed').value = dailySeed();
  const seed = $('seed').value.trim() || 'first-light'; $('seed').value = seed;
  game = createGame({ mode, difficulty: $('difficulty').value, seed });
  accumulator = 0; flashUntil = 0;
  $('mission-title').innerHTML = mode === 'classic' ? 'Find a line.<br>Make it last.' : 'Match. Collect.<br>Switch. Repeat.';
  $('mission-copy').textContent = mode === 'classic'
    ? 'The original rhythm. Collect energy, grow your snake, and leave yourself a way out. No gates or phase matching.'
    : 'Mint circles. Amber diamonds. Your head must match the energy and the gate you enter.';
  $('seed-note').textContent = mode === 'daily'
    ? 'UTC date seed. Standard pace. Best saved per day.'
    : 'Same seed + pace + moves = same run.';
  updateUI();
}

function updateUI() {
  const palette = colors(), active = ['running', 'paused'].includes(game.status);
  $('score').textContent = String(game.score).padStart(4, '0');
  $('best').textContent = String(Math.max(profile.bests[scoreKey(game)] || 0, game.score)).padStart(4, '0');
  $('phase-label').textContent = game.phase === 0 ? '● MINT' : '◆ AMBER';
  $('phase-label').style.color = palette[game.phase];
  const targetPhase = game.food?.phase ?? 0;
  $('target-icon').textContent = targetPhase === 0 ? '●' : '◆';
  $('target-icon').style.color = palette[targetPhase];
  $('target-label').textContent = game.food ? `${targetPhase === 0 ? 'Mint' : 'Amber'} · +100` : 'Circuit complete';
  $('collected').textContent = `${game.collected} collected`;
  $('pause').disabled = !active;
  $('pause').firstChild.textContent = game.status === 'paused' ? 'Resume ' : 'Pause ';
  $('difficulty').disabled = active || mode === 'daily';
  $('seed').disabled = active || mode === 'daily'; $('shuffle').disabled = active || mode === 'daily';
  document.querySelectorAll('[data-mode]').forEach(button => {
    button.disabled = active; button.classList.toggle('active', button.dataset.mode === mode);
    button.setAttribute('aria-pressed', String(button.dataset.mode === mode));
  });
  $('shift').disabled = game.status !== 'running' || mode === 'classic';
  $('shift-touch').disabled = $('shift').disabled;
  document.querySelectorAll('[data-direction]').forEach(button => { button.disabled = game.status !== 'running'; });
  $('contrast').textContent = `High contrast: ${profile.contrast ? 'on' : 'off'}`;
  $('contrast').setAttribute('aria-pressed', String(profile.contrast));
  const status = { ready: 'Ready when you are', running: 'Circuit live · keep moving', paused: 'Paused · take your time', over: 'Run complete · try a new line', won: 'Circuit complete · beautifully done' };
  const dot = document.createElement('span'); dot.className = 'status-dot';
  $('run-status').replaceChildren(dot, status[game.status]);
  $('overlay').hidden = game.status === 'running';
  if (game.status === 'ready') {
    $('overlay-tag').textContent = mode === 'daily' ? `DAILY CIRCUIT / ${game.seed.slice(6)}` : 'THE CIRCUIT IS YOURS';
    $('overlay-title').textContent = 'Make your move.';
    $('overlay-copy').innerHTML = mode === 'classic' ? 'Collect energy. Leave a way out.<br>See how far you can go.' : 'Collect matching energy.<br>Shift before the gates.';
    $('start').firstChild.textContent = 'Start run ';
  } else if (game.status === 'paused') {
    $('overlay-tag').textContent = 'A MOMENT BETWEEN MOVES';
    $('overlay-title').textContent = 'Take a breath.';
    $('overlay-copy').textContent = 'Your route will be right here.';
    $('start').firstChild.textContent = 'Keep going ';
  } else if (['over', 'won'].includes(game.status)) {
    $('overlay-tag').textContent = `${game.collected} ENERGY / ${String(game.score).padStart(4, '0')} POINTS`;
    $('overlay-title').textContent = game.status === 'won' ? 'What a circuit.' : 'One more turn?';
    $('overlay-copy').textContent = game.reason;
    $('start').firstChild.textContent = 'Try again ';
  }
  $('overlay-hint').textContent = mode === 'classic' ? 'ARROWS TO TURN · P TO PAUSE' : 'ARROWS TO TURN · SPACE TO SHIFT';
}

function begin() {
  if (['over', 'won'].includes(game.status)) newGame();
  if (game.status === 'paused') togglePause(game);
  else startGame(game);
  accumulator = 0; updateUI();
  $('game').scrollIntoView({ block: 'start', behavior: 'instant' });
  canvas.focus({ preventScroll: true });
  announce(`${modeNames[mode]} started. ${mode === 'classic' ? 'Collect energy.' : 'Mint phase. Match energy and gates.'}`);
}
function pause() {
  if (!['running', 'paused'].includes(game.status)) return;
  togglePause(game); accumulator = 0; updateUI();
  if (game.status === 'paused') { $('start').focus({ preventScroll: true }); announce('Game paused.'); }
  else { canvas.focus({ preventScroll: true }); announce('Game resumed.'); }
}
function shift() {
  if (shiftPhase(game)) { updateUI(); announce(`${game.phase ? 'Amber' : 'Mint'} phase.`); }
}

function rounded(x, y, size, radius, fill) {
  ctx.fillStyle = fill; ctx.beginPath(); ctx.roundRect(x, y, size, size, radius); ctx.fill();
}
function diamond(x, y, radius, fill) {
  ctx.fillStyle = fill; ctx.beginPath();
  ctx.moveTo(x, y - radius); ctx.lineTo(x + radius, y); ctx.lineTo(x, y + radius); ctx.lineTo(x - radius, y); ctx.closePath(); ctx.fill();
}
function draw(now) {
  const palette = colors();
  ctx.clearRect(0, 0, width, width);
  ctx.fillStyle = '#111c22'; ctx.fillRect(0, 0, width, width);
  ctx.strokeStyle = profile.contrast ? '#35454c' : '#1c2b32'; ctx.lineWidth = .7;
  ctx.beginPath();
  for (let n = 0; n <= SIZE; n++) {
    ctx.moveTo(n * cell, 0); ctx.lineTo(n * cell, width);
    ctx.moveTo(0, n * cell); ctx.lineTo(width, n * cell);
  }
  ctx.stroke();
  for (const point of game.walls) {
    const x = point.x * cell, y = point.y * cell;
    rounded(x + 3, y + 3, cell - 6, 3, profile.contrast ? '#71848d' : '#3a4c55');
    ctx.fillStyle = '#71838b'; ctx.fillRect(x + 8, y + 8, cell - 16, 2);
  }
  for (const point of game.gates) {
    const x = point.x * cell, y = point.y * cell, color = palette[point.phase];
    ctx.globalAlpha = game.phase === point.phase ? .16 : .32;
    rounded(x + 2, y + 2, cell - 4, 3, color); ctx.globalAlpha = 1;
    ctx.fillStyle = color; ctx.fillRect(x + 3, y + 4, 2, cell - 8); ctx.fillRect(x + cell - 5, y + 4, 2, cell - 8);
    if (point.phase === 0) { ctx.beginPath(); ctx.arc(x + cell / 2, y + cell / 2, 3, 0, Math.PI * 2); ctx.fill(); }
    else diamond(x + cell / 2, y + cell / 2, 4, color);
  }
  if (game.food) {
    const x = (game.food.x + .5) * cell, y = (game.food.y + .5) * cell, color = palette[game.food.phase];
    const pulse = reducedMotion.matches ? 0 : Math.sin(now / 260) * 1.2;
    ctx.globalAlpha = .12; ctx.beginPath(); ctx.arc(x, y, 14 + pulse, 0, Math.PI * 2); ctx.fillStyle = color; ctx.fill(); ctx.globalAlpha = 1;
    if (game.food.phase === 0) { ctx.beginPath(); ctx.arc(x, y, 6 + pulse / 3, 0, Math.PI * 2); ctx.fillStyle = color; ctx.fill(); }
    else diamond(x, y, 8 + pulse / 3, color);
    ctx.fillStyle = '#efffdd'; ctx.fillRect(x - 2, y - 3, 3, 2);
  }
  // Draw tail first so the face stays legible at tight corners.
  for (let index = game.snake.length - 1; index >= 0; index--) {
    const point = game.snake[index], x = point.x * cell, y = point.y * cell;
    ctx.globalAlpha = index === 0 ? 1 : Math.max(.45, 1 - index * .018);
    rounded(x + 2, y + 2, cell - 4, index === 0 ? 7 : 4, palette[index === 0 ? game.phase : point.phase]);
    ctx.globalAlpha = 1;
    if (index === 0) {
      const eyePairs = { right: [[18, 8], [18, 18]], left: [[7, 8], [7, 18]], up: [[8, 7], [18, 7]], down: [[8, 18], [18, 18]] };
      ctx.fillStyle = '#13221c';
      for (const [ex, ey] of eyePairs[game.direction]) ctx.fillRect(x + ex, y + ey, 3, 3);
    } else if (point.phase === 1) {
      ctx.fillStyle = '#162126'; ctx.globalAlpha = .35; ctx.fillRect(x + 12, y + 12, 4, 4); ctx.globalAlpha = 1;
    }
  }
  if (now < flashUntil && !reducedMotion.matches) { ctx.fillStyle = '#b3f8760c'; ctx.fillRect(0, 0, width, width); }
}

function frame(now) {
  const delta = previousFrame ? Math.min(now - previousFrame, 250) : 0; previousFrame = now;
  if (game.status === 'running') {
    accumulator += delta;
    while (accumulator >= tickDuration(game) && game.status === 'running') {
      accumulator -= tickDuration(game);
      const result = step(game);
      if (result.type === 'collect') { flashUntil = now + 100; announce(`${game.score} points. Next energy: ${game.food.phase ? 'amber' : 'mint'}.`); }
      if (['crash', 'win'].includes(result.type)) {
        recordRun(profile, game); persist(); renderRecords();
        announce(`${game.reason} Final score ${game.score}.`);
      }
      if (result.type !== 'move') updateUI();
    }
  }
  draw(now); requestAnimationFrame(frame);
}

$('start').addEventListener('click', begin);
$('pause').addEventListener('click', pause);
$('restart').addEventListener('click', () => { newGame(); $('start').focus({ preventScroll: true }); announce('New run ready. Press Start run.'); });
$('shift').addEventListener('click', () => { shift(); canvas.focus({ preventScroll: true }); });
$('shift-touch').addEventListener('click', shift);
document.querySelectorAll('[data-direction]').forEach(button => button.addEventListener('click', () => queueDirection(game, button.dataset.direction)));
document.querySelectorAll('[data-mode]').forEach(button => button.addEventListener('click', () => {
  if (mode === 'daily') $('seed').value = 'first-light';
  mode = button.dataset.mode; newGame();
}));
$('difficulty').addEventListener('change', () => { profile.difficulty = $('difficulty').value; persist(); newGame(); });
$('seed').addEventListener('change', newGame);
$('shuffle').addEventListener('click', () => {
  const values = new Uint32Array(1); window.crypto.getRandomValues(values);
  $('seed').value = `route-${values[0].toString(36)}`; newGame();
});
$('contrast').addEventListener('click', () => {
  profile.contrast = !profile.contrast; document.body.classList.toggle('high-contrast', profile.contrast); persist(); updateUI();
});
$('clear-records').addEventListener('click', () => {
  const cleared = clearProfile(storage);
  // Keep the visual preferences in memory; clearing removes all persisted data.
  profile = { ...emptyProfile(), difficulty: profile.difficulty, contrast: profile.contrast };
  renderRecords(); updateUI();
  if (cleared) toast('Local records cleared.');
  else { $('storage-note').textContent = 'Could not clear browser storage. Clear site data in browser settings.'; toast('Storage is blocked. Only this visit’s records were cleared.'); }
});
$('fullscreen').addEventListener('click', async () => {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await $('game').requestFullscreen();
  } catch { toast('Fullscreen is unavailable in this browser.'); }
});

const turnKeys = { ArrowUp: 'up', ArrowRight: 'right', ArrowDown: 'down', ArrowLeft: 'left', w: 'up', d: 'right', s: 'down', a: 'left' };
document.addEventListener('keydown', event => {
  if (event.ctrlKey || event.metaKey || event.altKey || event.target.closest('input, select, textarea')) return;
  const pressed = event.key.length === 1 ? event.key.toLowerCase() : event.key;
  if (turnKeys[pressed] && game.status === 'running') { event.preventDefault(); if (!event.repeat) queueDirection(game, turnKeys[pressed]); }
  else if (pressed === ' ' && !event.target.closest('button, a')) { event.preventDefault(); if (!event.repeat) shift(); }
  else if (pressed === 'p' || pressed === 'Escape') { if (!event.repeat) pause(); }
  else if (pressed === 'r' && !event.repeat) { newGame(); $('start').focus({ preventScroll: true }); }
  else if (pressed === 'Enter' && event.target === canvas && game.status !== 'running') { event.preventDefault(); begin(); }
});
document.addEventListener('visibilitychange', () => { if (document.hidden && game.status === 'running') pause(); });
window.addEventListener('blur', () => { if (game.status === 'running') pause(); });

renderRecords(); newGame(); requestAnimationFrame(frame);

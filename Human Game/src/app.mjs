import { SIZE, createGame, startGame, queueDirection, shiftPhase, togglePause, step, tickDuration, scoreKey, dailySeed } from './engine.mjs';
import { loadProfile, saveProfile, recordRun } from './storage.mjs';
import { LEVELS, getLevel, availableLevelIds, recommendedLevelId, medalFor, MEDAL_NAMES } from './levels.mjs';
import { createAudio } from './audio.mjs';

const $ = id => document.getElementById(id);
const canvas = $('board'), ctx = canvas.getContext('2d');
const width = 672, cell = width / SIZE;
const ratio = Math.min(window.devicePixelRatio || 1, 2);
canvas.width = width * ratio; canvas.height = width * ratio;
ctx.scale(ratio, ratio);
let storage;
try { storage = window.localStorage; }
catch { storage = { getItem: () => null, setItem: () => { throw new Error('Storage unavailable'); }, removeItem: () => { throw new Error('Storage unavailable'); } }; }
let profile = loadProfile(storage), mode = 'campaign';
let levelId = recommendedLevelId(profile.campaign), tutorialPrompt = null;
let game, accumulator = 0, previousFrame = 0, flashUntil = 0, toastTimeout;
const audio = createAudio();
const modeNames = { phase: 'Phase run', classic: 'Classic', daily: 'Daily circuit', campaign: 'Campaign' };
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
  const saved = saveProfile(storage, profile);
  if (!saved) $('storage-note').textContent = 'Storage unavailable. Scores last for this visit only.';
  return saved;
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
    const stage = run.mode === 'campaign' ? LEVELS.find(level => level.id === run.levelId) : null;
    label.append(number, stage?.name || modeNames[run.mode]);
    const score = document.createElement('strong'); score.textContent = String(run.score).padStart(4, '0');
    item.append(label, score); list.append(item);
  });
}

function createStageButtons() {
  for (const [index, level] of LEVELS.entries()) {
    const button = document.createElement('button');
    button.className = 'stage'; button.dataset.stage = level.id;
    const number = document.createElement('span'); number.className = 'stage-number'; number.textContent = String(index + 1).padStart(2, '0');
    const title = document.createElement('strong'); title.textContent = level.name;
    const medal = document.createElement('span'); medal.className = 'stage-medal';
    button.append(number, title, medal);
    button.addEventListener('click', () => { levelId = level.id; newGame(); });
    $('stages').append(button);
  }
}

function updateStages(active) {
  const available = availableLevelIds(profile.campaign);
  const completed = LEVELS.filter(level => profile.campaign[level.id]).length;
  $('campaign-count').textContent = `${completed} / ${LEVELS.length} COMPLETE`;
  $('reset-campaign').disabled = active || completed === 0;
  document.querySelectorAll('[data-stage]').forEach(button => {
    const id = button.dataset.stage, result = profile.campaign[id], unlocked = available.includes(id);
    button.disabled = active || !unlocked;
    button.classList.toggle('selected', id === levelId);
    button.classList.toggle('completed', Boolean(result));
    button.setAttribute('aria-pressed', String(id === levelId));
    button.title = result ? `${MEDAL_NAMES[result.medal]} · best ${result.ticks} moves`
      : unlocked ? `Play ${getLevel(id).name}` : 'Complete the previous stage to unlock';
    button.querySelector('.stage-medal').textContent = result ? '◆'.repeat(result.medal) : unlocked ? '↗' : '—';
    button.setAttribute('aria-label', `${getLevel(id).name}, ${result ? MEDAL_NAMES[result.medal] : unlocked ? 'available' : 'locked'}`);
  });
}

function newGame() {
  if (mode === 'daily') $('seed').value = dailySeed();
  const seed = $('seed').value.trim() || 'first-light'; $('seed').value = seed;
  game = createGame({ mode, difficulty: $('difficulty').value, seed, levelId });
  accumulator = 0; flashUntil = 0; tutorialPrompt = null;
  const level = mode === 'campaign' ? getLevel(levelId) : null;
  $('campaign-section').hidden = !level;
  $('seed-setting').hidden = Boolean(level);
  $('objective-bar').hidden = !level;
  $('mission-title').textContent = level?.name || (mode === 'classic' ? 'Find a line. Make it last.' : 'Match. Collect. Switch. Repeat.');
  $('mission-copy').textContent = level?.description || (mode === 'classic'
    ? 'The original rhythm. Collect energy, grow your snake, and leave yourself a way out. No gates or phase matching.'
    : 'Mint circles. Amber diamonds. Your head must match the energy and the gate you enter.');
  $('seed-note').textContent = mode === 'daily'
    ? 'UTC date seed. Standard pace. Best saved per day.'
    : 'Same seed + pace + moves = same run.';
  updateUI();
}

function updateUI() {
  const palette = colors(), active = ['running', 'paused'].includes(game.status);
  const level = mode === 'campaign' ? getLevel(levelId) : null;
  $('score').textContent = String(game.score).padStart(4, '0');
  $('best').textContent = String(Math.max(profile.bests[scoreKey(game)] || 0, game.score)).padStart(4, '0');
  $('phase-label').textContent = game.phase === 0 ? '● MINT' : '◆ AMBER';
  $('phase-label').style.color = palette[game.phase];
  const targetPhase = game.food?.phase ?? 0;
  $('target-icon').textContent = targetPhase === 0 ? '●' : '◆';
  $('target-icon').style.color = palette[targetPhase];
  $('target-label').textContent = game.food ? `${targetPhase === 0 ? 'Mint' : 'Amber'} · +100` : 'Circuit complete';
  $('collected').textContent = level ? `${game.collected} / ${game.target}` : `${game.collected} collected`;
  $('move-count').textContent = String(game.ticks);
  $('stage-progress').max = game.target || 1; $('stage-progress').value = game.collected;
  $('objective-name').textContent = level ? `${LEVELS.indexOf(level) + 1} / ${LEVELS.length} · ${level.name}` : '';
  $('objective-detail').textContent = level ? `${game.collected}/${game.target} energy · Gold ≤ ${level.par} moves` : '';
  $('stage-tip').textContent = level?.hints?.[game.collected] || (level ? `Collect all ${game.target} energy cells. Complete the stage to unlock the next circuit.` : '');
  updateStages(active);
  $('pause').disabled = !active;
  $('pause').firstChild.textContent = game.status === 'paused' ? 'Resume ' : 'Pause ';
  $('difficulty').disabled = active || mode === 'daily';
  $('seed').disabled = active || mode === 'daily' || Boolean(level); $('shuffle').disabled = $('seed').disabled;
  document.querySelectorAll('[data-mode]').forEach(button => {
    button.disabled = active; button.classList.toggle('active', button.dataset.mode === mode);
    button.setAttribute('aria-pressed', String(button.dataset.mode === mode));
  });
  $('shift').disabled = game.status !== 'running' || mode === 'classic';
  $('shift-touch').disabled = $('shift').disabled;
  document.querySelectorAll('[data-direction]').forEach(button => { button.disabled = game.status !== 'running'; });
  $('contrast').textContent = `High contrast: ${profile.contrast ? 'on' : 'off'}`;
  $('contrast').setAttribute('aria-pressed', String(profile.contrast));
  $('sound').textContent = `Sound: ${profile.sound ? 'on' : 'off'}`;
  $('sound').setAttribute('aria-pressed', String(profile.sound));
  const status = { ready: 'Ready when you are', running: 'Circuit live · keep moving', paused: 'Paused · take your time', over: 'Run complete · try a new line', won: 'Circuit complete · beautifully done' };
  const dot = document.createElement('span'); dot.className = 'status-dot';
  $('run-status').replaceChildren(dot, status[game.status]);
  $('overlay').hidden = game.status === 'running';
  if (game.status === 'ready') {
    $('overlay-tag').textContent = level?.label || (mode === 'daily' ? `DAILY CIRCUIT / ${game.seed.slice(6)}` : 'THE CIRCUIT IS YOURS');
    $('overlay-title').textContent = level?.name || 'Make your move.';
    $('overlay-copy').textContent = level?.hints?.[0] || level?.description || (mode === 'classic' ? 'Collect energy. Leave a way out. See how far you can go.' : 'Collect matching energy. Shift before the gates.');
    $('start').firstChild.textContent = level ? (level.id === 'first-light' ? 'Start lesson ' : 'Start stage ') : 'Start run ';
  } else if (game.status === 'paused') {
    $('overlay-tag').textContent = tutorialPrompt ? `LESSON ${game.collected + 1} / 3` : 'A MOMENT BETWEEN MOVES';
    $('overlay-title').textContent = tutorialPrompt ? (game.collected === 1 ? 'Switch your phase.' : 'Thread the gate.') : 'Take a breath.';
    $('overlay-copy').textContent = tutorialPrompt || 'Your route will be right here.';
    $('start').firstChild.textContent = tutorialPrompt ? 'Continue lesson ' : 'Keep going ';
  } else if (['over', 'won'].includes(game.status)) {
    const campaignWin = level && game.status === 'won';
    $('overlay-tag').textContent = campaignWin ? `${'◆'.repeat(medalFor(game))} ${MEDAL_NAMES[medalFor(game)].toUpperCase()} ROUTE` : `${game.collected} ENERGY / ${String(game.score).padStart(4, '0')} POINTS`;
    $('overlay-title').textContent = campaignWin ? (level === LEVELS.at(-1) ? 'All circuits lit.' : 'Circuit complete.') : game.status === 'won' ? 'What a circuit.' : 'One more turn?';
    $('overlay-copy').textContent = campaignWin ? `${game.ticks} moves. ${game.score} points. ${level === LEVELS.at(-1) ? 'Replay any stage to improve your medals.' : 'The next stage is unlocked.'}` : game.reason;
    $('start').firstChild.textContent = campaignWin ? (level === LEVELS.at(-1) ? 'Replay finale ' : 'Next stage ') : 'Try again ';
  }
  $('overlay-hint').textContent = mode === 'classic' ? 'ARROWS TO TURN · P TO PAUSE' : 'ARROWS TO TURN · SPACE TO SHIFT';
}

function begin() {
  if (mode === 'campaign' && game.status === 'won') {
    levelId = LEVELS[Math.min(LEVELS.findIndex(level => level.id === levelId) + 1, LEVELS.length - 1)].id;
  }
  if (['over', 'won'].includes(game.status)) newGame();
  tutorialPrompt = null;
  if (game.status === 'paused') togglePause(game);
  else startGame(game);
  accumulator = 0; updateUI();
  $('game').scrollIntoView({ block: 'start', behavior: 'instant' });
  canvas.focus({ preventScroll: true });
  if (profile.sound) void audio.unlock();
  announce(`${modeNames[mode]} started. ${mode === 'classic' ? 'Collect energy.' : 'Mint phase. Match energy and gates.'}`);
}
function pause() {
  if (!['running', 'paused'].includes(game.status)) return;
  togglePause(game); accumulator = 0; updateUI();
  if (game.status === 'running') tutorialPrompt = null;
  if (game.status === 'paused') { $('start').focus({ preventScroll: true }); announce('Game paused.'); }
  else { canvas.focus({ preventScroll: true }); announce('Game resumed.'); }
}
function shift() {
  if (shiftPhase(game)) { if (profile.sound) audio.play('shift'); updateUI(); announce(`${game.phase ? 'Amber' : 'Mint'} phase.`); }
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
  if (mode === 'campaign') {
    const upcoming = getLevel(levelId).foods.slice(game.collected + 1);
    ctx.font = '10px monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    upcoming.forEach((point, index) => {
      const x = (point.x + .5) * cell, y = (point.y + .5) * cell;
      ctx.globalAlpha = .35; ctx.strokeStyle = palette[point.phase]; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.arc(x, y, 9, 0, Math.PI * 2); ctx.stroke();
      ctx.fillStyle = palette[point.phase]; ctx.fillText(String(game.collected + index + 2), x, y);
    });
    ctx.globalAlpha = 1;
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
  if (game.food) {
    const x = (game.food.x + .5) * cell, y = (game.food.y + .5) * cell, color = palette[game.food.phase];
    const pulse = reducedMotion.matches ? 0 : Math.sin(now / 260) * 1.2;
    ctx.globalAlpha = .12; ctx.beginPath(); ctx.arc(x, y, 14 + pulse, 0, Math.PI * 2); ctx.fillStyle = color; ctx.fill(); ctx.globalAlpha = 1;
    if (game.food.phase === 0) { ctx.beginPath(); ctx.arc(x, y, 6 + pulse / 3, 0, Math.PI * 2); ctx.fillStyle = color; ctx.fill(); }
    else diamond(x, y, 8 + pulse / 3, color);
    ctx.fillStyle = '#efffdd'; ctx.fillRect(x - 2, y - 3, 3, 2);
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
      if (result.type === 'collect') {
        flashUntil = now + 100; if (profile.sound) audio.play('collect');
        announce(`${game.score} points. Next energy: ${game.food.phase ? 'amber' : 'mint'}.`);
        if (mode === 'campaign' && levelId === 'first-light' && !profile.campaign['first-light']) {
          tutorialPrompt = getLevel(levelId).hints[game.collected];
          togglePause(game); accumulator = 0;
          announce(tutorialPrompt);
        }
      }
      if (['crash', 'win'].includes(result.type)) {
        recordRun(profile, game); persist(); renderRecords();
        if (profile.sound) audio.play(result.type === 'win' ? 'win' : 'crash');
        announce(`${game.reason} Final score ${game.score}.`);
      }
      if (result.type !== 'move') updateUI();
      else $('move-count').textContent = String(game.ticks);
      if (tutorialPrompt) $('start').focus({ preventScroll: true });
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
  profile.runs = []; const saved = persist(); renderRecords();
  toast(saved ? 'Run history cleared. Your bests and campaign progress are kept.' : 'History cleared for this visit. Browser storage could not be updated.');
});
$('reset-campaign').addEventListener('click', () => {
  if (!window.confirm('Reset campaign progress and medals? Arcade records and settings will be kept.')) return;
  profile.campaign = {};
  for (const key of Object.keys(profile.bests)) if (key.startsWith('campaign:')) delete profile.bests[key];
  levelId = LEVELS[0].id; const saved = persist(); newGame();
  toast(saved ? 'Campaign reset. First Light is ready.' : 'Campaign reset for this visit. Browser storage could not be updated.');
});
$('sound').addEventListener('click', async () => {
  if (!profile.sound && !await audio.unlock()) { toast('Sound is unavailable in this browser.'); return; }
  profile.sound = !profile.sound; persist(); updateUI();
  if (profile.sound) audio.play('collect');
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

let swipe = null;
canvas.addEventListener('pointerdown', event => {
  if (game.status !== 'running' || event.pointerType === 'mouse') return;
  swipe = { x: event.clientX, y: event.clientY, id: event.pointerId };
  try { canvas.setPointerCapture(event.pointerId); } catch { /* Continue without capture if the platform declines it. */ }
});
canvas.addEventListener('pointerup', event => {
  if (!swipe || swipe.id !== event.pointerId) return;
  const dx = event.clientX - swipe.x, dy = event.clientY - swipe.y; swipe = null;
  if (Math.max(Math.abs(dx), Math.abs(dy)) < 16) return;
  queueDirection(game, Math.abs(dx) > Math.abs(dy) ? (dx > 0 ? 'right' : 'left') : (dy > 0 ? 'down' : 'up'));
});
canvas.addEventListener('pointercancel', () => { swipe = null; });

createStageButtons(); renderRecords(); newGame(); requestAnimationFrame(frame);

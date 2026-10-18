import { LEVELS, medalFor } from './levels.mjs';
import { scoreKey } from './engine.mjs';

const STORAGE_KEY = 'switchback:v1'; // Keep existing v0.1 scores when upgrading.
export const emptyProfile = () => ({ bests: {}, runs: [], campaign: {}, difficulty: 'standard', contrast: false, sound: false });

export function loadProfile(storage) {
  try {
    const raw = JSON.parse(storage.getItem(STORAGE_KEY) || 'null');
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return emptyProfile();
    const bests = {};
    for (const [key, score] of Object.entries(raw.bests || {}).slice(0, 300)) {
      if (/^(phase|classic|daily|campaign):/.test(key) && Number.isSafeInteger(score) && score >= 0) bests[key] = score;
    }
    const campaign = {};
    for (const level of LEVELS) {
      const result = raw.campaign?.[level.id];
      if (result && Number.isInteger(result.medal) && result.medal >= 1 && result.medal <= 3
        && Number.isSafeInteger(result.ticks) && result.ticks > 0) {
        campaign[level.id] = { medal: result.medal, ticks: result.ticks };
      }
    }
    return {
      bests, campaign,
      runs: Array.isArray(raw.runs) ? raw.runs.filter(run => run && typeof run === 'object'
        && typeof run.mode === 'string' && ['phase', 'classic', 'daily', 'campaign'].includes(run.mode)
        && Number.isSafeInteger(run.score) && run.score >= 0).slice(0, 5) : [],
      difficulty: ['cruise', 'standard', 'rush'].includes(raw.difficulty) ? raw.difficulty : 'standard',
      contrast: raw.contrast === true,
      sound: raw.sound === true,
    };
  } catch { return emptyProfile(); }
}

export function saveProfile(storage, profile) {
  try { storage.setItem(STORAGE_KEY, JSON.stringify(profile)); return true; }
  catch { return false; }
}

export function clearProfile(storage) {
  try { storage.removeItem(STORAGE_KEY); return true; }
  catch { return false; }
}

export function recordRun(profile, game, date = new Date()) {
  const key = scoreKey(game);
  profile.bests[key] = Math.max(profile.bests[key] || 0, game.score);
  profile.runs.unshift({ mode: game.mode, score: game.score, collected: game.collected, levelId: game.levelId, date: date.toISOString() });
  profile.runs = profile.runs.slice(0, 5);
  if (game.mode === 'campaign' && game.status === 'won') {
    profile.campaign ||= {};
    const previous = profile.campaign[game.levelId];
    profile.campaign[game.levelId] = {
      medal: Math.max(previous?.medal || 0, medalFor(game)),
      ticks: Math.min(previous?.ticks || Infinity, game.ticks),
    };
  }
  return profile;
}

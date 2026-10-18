const STORAGE_KEY = 'switchback:v1';
export const emptyProfile = () => ({ bests: {}, runs: [], difficulty: 'standard', contrast: false });

export function loadProfile(storage) {
  try {
    const raw = JSON.parse(storage.getItem(STORAGE_KEY) || 'null');
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return emptyProfile();
    const bests = {};
    for (const [key, score] of Object.entries(raw.bests || {}).slice(0, 300)) {
      if (/^(phase|classic|daily):/.test(key) && Number.isSafeInteger(score) && score >= 0) bests[key] = score;
    }
    return {
      bests,
      runs: Array.isArray(raw.runs) ? raw.runs.filter(run => run && typeof run === 'object'
        && typeof run.mode === 'string' && ['phase', 'classic', 'daily'].includes(run.mode)
        && Number.isSafeInteger(run.score) && run.score >= 0).slice(0, 5) : [],
      difficulty: ['cruise', 'standard', 'rush'].includes(raw.difficulty) ? raw.difficulty : 'standard',
      contrast: raw.contrast === true,
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
  const key = `${game.mode}:${game.difficulty}${game.mode === 'daily' ? `:${game.seed}` : ''}`;
  profile.bests[key] = Math.max(profile.bests[key] || 0, game.score);
  profile.runs.unshift({ mode: game.mode, score: game.score, collected: game.collected, date: date.toISOString() });
  profile.runs = profile.runs.slice(0, 5);
  return profile;
}

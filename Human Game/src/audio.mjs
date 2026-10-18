// Original synthesized feedback. No downloaded sound files or music.
export function createAudio() {
  let context;
  const unlock = async () => {
    try {
      const Constructor = globalThis.AudioContext || globalThis.webkitAudioContext;
      if (!Constructor) return false;
      context ||= new Constructor();
      if (context.state === 'suspended') await context.resume();
      return context.state === 'running';
    } catch { return false; }
  };
  const play = kind => {
    if (!context || context.state !== 'running') return;
    const notes = { shift: [330, 440], collect: [523, 784], crash: [196, 98], win: [523, 659, 784, 1047] }[kind] || [];
    try {
      notes.forEach((frequency, index) => {
        const time = context.currentTime + index * .065;
        const oscillator = context.createOscillator(), gain = context.createGain();
        oscillator.type = 'sine'; oscillator.frequency.setValueAtTime(frequency, time);
        gain.gain.setValueAtTime(0, time);
        gain.gain.linearRampToValueAtTime(.055, time + .008);
        gain.gain.exponentialRampToValueAtTime(.001, time + .13);
        oscillator.connect(gain); gain.connect(context.destination);
        oscillator.start(time); oscillator.stop(time + .15);
        oscillator.onended = () => { oscillator.disconnect(); gain.disconnect(); };
      });
    } catch { /* Sound never controls whether a game can run. */ }
  };
  return { unlock, play };
}

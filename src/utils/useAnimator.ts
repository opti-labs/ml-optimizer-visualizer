import { useState, useRef, useCallback, useEffect } from 'react';

export type RunStatus = 'idle' | 'running' | 'paused' | 'done';

/**
 * Shared run/step/pause/reset loop for the optimizer pages.
 *
 * `stepOnce` performs exactly one optimization step (mutating the caller's refs
 * and flushing React state) and returns true when finished. It is stored in a
 * ref so the interval always calls the latest closure — and crucially the
 * interval calls it only ONCE per tick, so React StrictMode's double-invocation
 * of render-phase functions can't double-advance the optimizer.
 */
export function useAnimator(stepOnce: () => boolean, onReset: () => void, speed: number) {
  const [status, setStatus] = useState<RunStatus>('idle');
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const stepRef = useRef(stepOnce);
  const resetRef = useRef(onReset);
  const speedRef = useRef(speed);
  stepRef.current = stepOnce;
  resetRef.current = onReset;
  speedRef.current = speed;

  const stop = useCallback(() => {
    if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null; }
  }, []);

  const start = useCallback(() => {
    stop();
    setStatus('running');
    intervalRef.current = setInterval(() => {
      const done = stepRef.current();
      if (done) { stop(); setStatus('done'); }
    }, speedRef.current);
  }, [stop]);

  const pause = useCallback(() => { stop(); setStatus('paused'); }, [stop]);

  const step = useCallback(() => {
    const done = stepRef.current();
    setStatus(done ? 'done' : 'paused');
  }, []);

  const reset = useCallback(() => {
    stop();
    resetRef.current();
    setStatus('idle');
  }, [stop]);

  useEffect(() => () => stop(), [stop]);

  return { status, start, pause, step, reset };
}

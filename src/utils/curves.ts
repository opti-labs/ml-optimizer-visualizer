import { EPS } from './math';

// ── Regression function families ─────────────────────────────────────────────
// All regression data lives on a common domain so every family (including the
// Poisson PMF, which is only defined for x >= 0) coexists and can be mixed.
export const X_MIN = 0;
export const X_MAX = 8;

export type RegFamily = 'linear' | 'sinusoidal' | 'exponential' | 'gaussian' | 'poisson';

export const FAMILY_LABEL: Record<RegFamily, string> = {
  linear: '線形 (y = ax + b)',
  sinusoidal: '三角関数 (y = a·sin(bx) + c)',
  exponential: '指数関数 (y = a·exp(bx))',
  gaussian: 'ガウス関数 (y = a·exp(−(x−b)²/2c²))',
  poisson: 'ポアソン分布 (y = a·λˣe⁻ᵏ/x!)',
};

export interface CurveParams {
  type: RegFamily;
  theta: number[];
}

// ── log Γ via Lanczos (for the continuous Poisson PMF) ───────────────────────
const LANCZOS = [
  676.5203681218851, -1259.1392167224028, 771.32342877765313,
  -176.61502916214059, 12.507343278686905, -0.13857109526572012,
  9.9843695780195716e-6, 1.5056327351493116e-7,
];

function logGamma(z: number): number {
  if (z < 0.5) {
    return Math.log(Math.PI / Math.sin(Math.PI * z)) - logGamma(1 - z);
  }
  z -= 1;
  let x = 0.99999999999980993;
  for (let i = 0; i < LANCZOS.length; i++) x += LANCZOS[i] / (z + i + 1);
  const t = z + LANCZOS.length - 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
}

// ── Evaluate a curve at x ────────────────────────────────────────────────────
export function evalCurve(p: CurveParams, x: number): number {
  switch (p.type) {
    case 'linear':
      return p.theta[0] * x + p.theta[1];
    case 'sinusoidal':
      return p.theta[0] * Math.sin(p.theta[1] * x) + p.theta[2];
    case 'exponential':
      return p.theta[0] * Math.exp(clampExp(p.theta[1] * x));
    case 'gaussian': {
      const [a, b, c] = p.theta;
      return a * Math.exp(-((x - b) ** 2) / (2 * Math.max(c * c, EPS)));
    }
    case 'poisson': {
      const [a, lam] = p.theta;
      if (x < 0) return 0;
      const l = Math.max(lam, EPS);
      return a * Math.exp(x * Math.log(l) - l - logGamma(x + 1));
    }
  }
}

function clampExp(v: number) {
  return Math.max(-30, Math.min(30, v));
}

export function sampleCurve(p: CurveParams, nPts = 120, xMin = X_MIN, xMax = X_MAX) {
  const xs: number[] = [], ys: number[] = [];
  for (let i = 0; i < nPts; i++) {
    const x = xMin + ((xMax - xMin) * i) / (nPts - 1);
    xs.push(x);
    ys.push(evalCurve({ type: p.type, theta: p.theta }, x));
  }
  return { x: xs, y: ys };
}

// ── Random initialization ────────────────────────────────────────────────────
// Deliberately "wrong" starting parameters (bounded to stay roughly on-screen)
// so the EM convergence process is actually visible step by step. For a mixture
// the per-component randomness also breaks symmetry between the two families.
function rand(lo: number, hi: number) {
  return lo + Math.random() * (hi - lo);
}

export function randomCurveParams(type: RegFamily): CurveParams {
  const mid = (X_MIN + X_MAX) / 2;
  switch (type) {
    case 'linear':
      return { type, theta: [rand(-1, 1), rand(-2, 3)] };
    case 'sinusoidal':
      return { type, theta: [rand(0.5, 2.0), rand(0.4, 2.6), rand(-0.5, 2.5)] };
    case 'exponential':
      return { type, theta: [rand(0.2, 1.0), rand(-0.2, 0.3)] };
    case 'gaussian':
      return { type, theta: [rand(1.0, 3.0), rand(X_MIN + 1, X_MAX - 1), rand(0.7, 2.2)] };
    case 'poisson':
      return { type, theta: [rand(4, 14), rand(mid - 2, mid + 2)] };
  }
}

// ── Weighted least-squares helpers ───────────────────────────────────────────

/** Weighted 1-D amplitude fit: a = Σ w·φ·y / Σ w·φ²  for model y ≈ a·φ(x). */
function fitAmplitude(phi: number[], ys: number[], w: number[]): number {
  let num = 0, den = 0;
  for (let i = 0; i < ys.length; i++) {
    num += w[i] * phi[i] * ys[i];
    den += w[i] * phi[i] * phi[i];
  }
  return den > EPS ? num / den : 0;
}

/** Weighted fit of y ≈ a·φ(x) + c (2 basis columns: φ and 1). Returns [a, c]. */
function fitAffine(phi: number[], ys: number[], w: number[]): [number, number] {
  let sww = 0, swp = 0, swpp = 0, swy = 0, swpy = 0;
  for (let i = 0; i < ys.length; i++) {
    sww += w[i];
    swp += w[i] * phi[i];
    swpp += w[i] * phi[i] * phi[i];
    swy += w[i] * ys[i];
    swpy += w[i] * phi[i] * ys[i];
  }
  const det = swpp * sww - swp * swp;
  if (Math.abs(det) < EPS) return [0, sww > EPS ? swy / sww : 0];
  const a = (swpy * sww - swp * swy) / det;
  const c = (swpp * swy - swp * swpy) / det;
  return [a, c];
}

function weightedSSE(p: CurveParams, xs: number[], ys: number[], w: number[]): number {
  let s = 0;
  for (let i = 0; i < xs.length; i++) {
    const r = ys[i] - evalCurve(p, xs[i]);
    s += w[i] * r * r;
  }
  return s;
}

function range(lo: number, hi: number, step: number): number[] {
  const out: number[] = [];
  for (let v = lo; v <= hi + 1e-9; v += step) out.push(Number(v.toFixed(6)));
  return out;
}

/**
 * Fit one family to (weighted) data. Nonlinear "inner" parameters are found by
 * a coarse grid search; the amplitude (and constant offset) are solved exactly
 * by weighted least squares for each grid point. This is deterministic and
 * numerically stable — no gradient-descent convergence issues.
 */
export function fitWeightedCurve(
  type: RegFamily,
  xs: number[],
  ys: number[],
  w: number[]
): CurveParams {
  switch (type) {
    case 'linear': {
      // exact weighted least squares for [a, b] in y = a·x + b
      const [a, b] = fitAffine(xs, ys, w);
      return { type, theta: [a, b] };
    }
    case 'sinusoidal': {
      let best: CurveParams = { type, theta: [1, 1, 0] };
      let bestSSE = Infinity;
      for (const bGrid of range(0.3, 3.0, 0.1)) {
        const phi = xs.map(x => Math.sin(bGrid * x));
        const [a, c] = fitAffine(phi, ys, w);
        const cand: CurveParams = { type, theta: [a, bGrid, c] };
        const sse = weightedSSE(cand, xs, ys, w);
        if (sse < bestSSE) { bestSSE = sse; best = cand; }
      }
      return best;
    }
    case 'exponential': {
      let best: CurveParams = { type, theta: [1, 0] };
      let bestSSE = Infinity;
      for (const bGrid of range(-1.2, 1.2, 0.1)) {
        const phi = xs.map(x => Math.exp(clampExp(bGrid * x)));
        const a = fitAmplitude(phi, ys, w);
        const cand: CurveParams = { type, theta: [a, bGrid] };
        const sse = weightedSSE(cand, xs, ys, w);
        if (sse < bestSSE) { bestSSE = sse; best = cand; }
      }
      return best;
    }
    case 'gaussian': {
      let best: CurveParams = { type, theta: [1, (X_MIN + X_MAX) / 2, 1] };
      let bestSSE = Infinity;
      for (const bGrid of range(X_MIN, X_MAX, 0.4)) {
        for (const cGrid of [0.4, 0.6, 0.9, 1.2, 1.6, 2.2]) {
          const phi = xs.map(x => Math.exp(-((x - bGrid) ** 2) / (2 * cGrid * cGrid)));
          const a = fitAmplitude(phi, ys, w);
          const cand: CurveParams = { type, theta: [a, bGrid, cGrid] };
          const sse = weightedSSE(cand, xs, ys, w);
          if (sse < bestSSE) { bestSSE = sse; best = cand; }
        }
      }
      return best;
    }
    case 'poisson': {
      let best: CurveParams = { type, theta: [1, 3] };
      let bestSSE = Infinity;
      for (const lam of range(0.5, 10, 0.25)) {
        const phi = xs.map(x => (x < 0 ? 0 : Math.exp(x * Math.log(Math.max(lam, EPS)) - lam - logGamma(x + 1))));
        const a = fitAmplitude(phi, ys, w);
        const cand: CurveParams = { type, theta: [a, lam] };
        const sse = weightedSSE(cand, xs, ys, w);
        if (sse < bestSSE) { bestSSE = sse; best = cand; }
      }
      return best;
    }
  }
}

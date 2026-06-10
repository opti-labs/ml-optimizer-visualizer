import type { DataPoint } from './dataGen';
import { randn } from './math';
import type { FnModel } from './functions';

// ── Data generation for the fit task ─────────────────────────────────────────
export function generateFitData(
  model: FnModel,
  trueTheta: number[],
  n: number,
  noise: number
): DataPoint[] {
  const [lo, hi] = model.domain;
  const pts: DataPoint[] = [];
  for (let i = 0; i < n; i++) {
    const x = lo + Math.random() * (hi - lo);
    pts.push({ x, y: model.eval(trueTheta, x) + randn() * noise });
  }
  pts.sort((a, b) => a.x - b.x);
  return pts;
}

// ── Adam gradient-descent fitter ─────────────────────────────────────────────
// Adam normalizes per-parameter step sizes, so a single learning rate works
// across the very different scales of a/b, frequency k, σ, mixture weights …
// while still diverging if the learning rate is pushed too high.

export interface FitState {
  theta: number[];
  m: number[];   // 1st moment
  v: number[];   // 2nd moment
  t: number;     // step count (bias correction)
}

export function initFitState(model: FnModel, data?: DataPoint[]): FitState {
  const raw = model.smartInit && data ? model.smartInit(data) : model.randomInit();
  const theta = model.clamp(raw);
  return { theta, m: theta.map(() => 0), v: theta.map(() => 0), t: 0 };
}

export function fitLoss(data: DataPoint[], model: FnModel, theta: number[]): number {
  let s = 0;
  for (const p of data) {
    const e = model.eval(theta, p.x) - p.y;
    s += e * e;
  }
  return s / data.length;
}

const B1 = 0.9, B2 = 0.999, EPS = 1e-8;

export function fitGDStep(
  data: DataPoint[],
  model: FnModel,
  state: FitState,
  lr: number
): { state: FitState; loss: number } {
  const n = data.length;
  const g = state.theta.map(() => 0);
  let loss = 0;

  for (const p of data) {
    const f = model.eval(state.theta, p.x);
    const err = f - p.y;
    loss += err * err;
    const gr = model.grad(state.theta, p.x);
    for (let j = 0; j < g.length; j++) g[j] += (2 / n) * err * gr[j];
  }
  loss /= n;

  const t = state.t + 1;
  const m = state.m.map((mj, j) => B1 * mj + (1 - B1) * g[j]);
  const v = state.v.map((vj, j) => B2 * vj + (1 - B2) * g[j] * g[j]);

  let theta = state.theta.map((th, j) => {
    const mhat = m[j] / (1 - Math.pow(B1, t));
    const vhat = v[j] / (1 - Math.pow(B2, t));
    return th - lr * mhat / (Math.sqrt(vhat) + EPS);
  });
  theta = model.clamp(theta);

  return { state: { theta, m, v, t }, loss };
}

// ── Numerical utilities ──────────────────────────────────────────────────────

export const EPS = 1e-10;

export function clamp(v: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, v));
}

/** Box-Muller transform: standard normal sample */
export function randn(): number {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

/** Sample from N(mu, sigma^2) */
export function randnScaled(mu: number, sigma: number) {
  return mu + sigma * randn();
}

/** Univariate Gaussian PDF */
export function gaussianPDF(x: number, mu: number, sigma2: number): number {
  const s2 = Math.max(sigma2, EPS);
  return (1 / Math.sqrt(2 * Math.PI * s2)) * Math.exp(-0.5 * ((x - mu) ** 2) / s2);
}

/** 2-D diagonal Gaussian PDF */
export function gaussian2D(
  x: number, y: number,
  mx: number, my: number,
  sx2: number, sy2: number
): number {
  const sx = Math.max(sx2, EPS);
  const sy = Math.max(sy2, EPS);
  return (
    (1 / (2 * Math.PI * Math.sqrt(sx * sy))) *
    Math.exp(-0.5 * (((x - mx) ** 2) / sx + ((y - my) ** 2) / sy))
  );
}

/** Log-sum-exp trick for numerical stability */
export function logSumExp(logs: number[]): number {
  const m = Math.max(...logs);
  if (!isFinite(m)) return -Infinity;
  return m + Math.log(logs.reduce((s, l) => s + Math.exp(l - m), 0));
}

/** Softmax */
export function softmax(logits: number[]): number[] {
  const m = Math.max(...logits);
  const exps = logits.map(l => Math.exp(l - m));
  const sum = exps.reduce((a, b) => a + b, 0) + EPS;
  return exps.map(e => e / sum);
}

/** Ellipse points for a 2-D Gaussian (diagonal covariance) */
export function ellipsePoints(
  mx: number, my: number,
  sx2: number, sy2: number,
  nSigma = 2,
  nPts = 60
): { x: number[]; y: number[] } {
  const sx = Math.sqrt(Math.max(sx2, EPS));
  const sy = Math.sqrt(Math.max(sy2, EPS));
  const xs: number[] = [], ys: number[] = [];
  for (let i = 0; i <= nPts; i++) {
    const t = (2 * Math.PI * i) / nPts;
    xs.push(mx + nSigma * sx * Math.cos(t));
    ys.push(my + nSigma * sy * Math.sin(t));
  }
  return { x: xs, y: ys };
}

/** MSE */
export function mse(preds: number[], targets: number[]): number {
  return preds.reduce((s, p, i) => s + (p - targets[i]) ** 2, 0) / preds.length;
}

/** Sigmoid */
export function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-clamp(x, -30, 30)));
}

/** ReLU */
export function relu(x: number): number {
  return Math.max(0, x);
}

/** tanh */
export function tanhFn(x: number): number {
  return Math.tanh(x);
}

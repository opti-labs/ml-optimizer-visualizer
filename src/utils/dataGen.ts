import { randn, randnScaled } from './math';
import { evalCurve, X_MIN, X_MAX } from './curves';
import type { CurveParams, RegFamily } from './curves';

export type DataPoint = { x: number; y: number; label?: number };

// User-facing selections: the 5 base families plus a 2-family mixture.
export type RegressionType = RegFamily | 'mixed';

export interface RegressionDataset {
  points: DataPoint[];
  /** Ground-truth families the data was generated from (1 for a single
   *  shape, 2 for a mixture). The EM mixture-of-regressions fits exactly
   *  these families so it can untangle the components. */
  families: RegFamily[];
}

// "True" generating parameters for each base family, all on the common
// domain [X_MIN, X_MAX] so any pair can be mixed coherently.
const TRUE_PARAMS: Record<RegFamily, CurveParams> = {
  linear: { type: 'linear', theta: [0.8, -1.0] },
  sinusoidal: { type: 'sinusoidal', theta: [1.6, 1.2, 1.5] },
  exponential: { type: 'exponential', theta: [0.2, 0.45] },
  gaussian: { type: 'gaussian', theta: [3.0, 3.0, 1.2] },
  poisson: { type: 'poisson', theta: [13.4, 3.0] },
};

const BASE_FAMILIES: RegFamily[] = ['linear', 'sinusoidal', 'exponential', 'gaussian', 'poisson'];

function randX(): number {
  return X_MIN + Math.random() * (X_MAX - X_MIN);
}

function pickTwoDistinct(): [RegFamily, RegFamily] {
  const a = BASE_FAMILIES[Math.floor(Math.random() * BASE_FAMILIES.length)];
  let b = a;
  while (b === a) b = BASE_FAMILIES[Math.floor(Math.random() * BASE_FAMILIES.length)];
  return [a, b];
}

// ── Regression data ────────────────────────────────────────────────────────

export function generateRegressionData(
  type: RegressionType,
  n: number,
  noise: number
): RegressionDataset {
  const pts: DataPoint[] = [];

  if (type === 'mixed') {
    // Mixed regression = a probabilistic blend of TWO DIFFERENT families
    // (e.g. a line and a Gaussian bump), not two parallel lines.
    const families = pickTwoDistinct();
    for (let i = 0; i < n; i++) {
      const x = randX();
      const comp = Math.random() < 0.5 ? 0 : 1;
      const y = evalCurve(TRUE_PARAMS[families[comp]], x) + randn() * noise;
      pts.push({ x, y, label: comp });
    }
    pts.sort((a, b) => a.x - b.x);
    return { points: pts, families };
  }

  // Single family
  const fam = type as RegFamily;
  for (let i = 0; i < n; i++) {
    const x = randX();
    pts.push({ x, y: evalCurve(TRUE_PARAMS[fam], x) + randn() * noise, label: 0 });
  }
  pts.sort((a, b) => a.x - b.x);
  return { points: pts, families: [fam] };
}

// ── Clustering / GMM data ──────────────────────────────────────────────────

export interface GMMTrueParams {
  means: [number, number][];
  varX: number[];
  varY: number[];
  weights: number[];
}

const CLUSTER_PALETTES: [number, number][][] = [
  [[-2, -2], [2, 2]],
  [[-3, 0], [3, 0], [0, 3]],
  [[-2, -2], [2, -2], [-2, 2], [2, 2]],
];

export function generateClusterData(
  k: number,
  n: number,
  noise: number
): { points: DataPoint[]; trueParams: GMMTrueParams } {
  const baseMeans = CLUSTER_PALETTES[Math.min(k - 2, 2)];
  const means: [number, number][] = baseMeans.map(([mx, my]) => [
    mx + randn() * 0.3,
    my + randn() * 0.3,
  ]);
  const varX = means.map(() => 0.4 + Math.random() * 0.4 + noise * noise);
  const varY = means.map(() => 0.4 + Math.random() * 0.4 + noise * noise);
  const rawW = means.map(() => 0.5 + Math.random());
  const wSum = rawW.reduce((a, b) => a + b, 0);
  const weights = rawW.map(w => w / wSum);

  const points: DataPoint[] = [];
  for (let i = 0; i < n; i++) {
    // sample component
    const r = Math.random();
    let cumW = 0, comp = 0;
    for (let c = 0; c < k; c++) {
      cumW += weights[c];
      if (r < cumW) { comp = c; break; }
    }
    points.push({
      x: randnScaled(means[comp][0], Math.sqrt(varX[comp])),
      y: randnScaled(means[comp][1], Math.sqrt(varY[comp])),
      label: comp,
    });
  }

  return { points, trueParams: { means, varX, varY, weights } };
}

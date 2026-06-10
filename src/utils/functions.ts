// ── Parametric function families for the "function fit" task ─────────────────
// Each model is fit to noisy data by gradient descent on its OWN parameters
// (the parameter vector θ). The user sets the TRUE θ to generate data, then
// watches gradient descent recover it from a random start.

export type FitFunctionType = 'line' | 'trig' | 'gaussian' | 'gmm';

export const FIT_LABEL: Record<FitFunctionType, string> = {
  line: '直線  y = a·x + b',
  trig: '三角関数  y = A·cos(kx) + B·sin(kx)',
  gaussian: 'ガウス分布  y = exp(−(x−μ)² / 2σ²)',
  gmm: '混合ガウス分布  y = Σ wᵢ·exp(−(x−μᵢ)²/2σᵢ²)',
};

export interface ParamField {
  label: string;
  min: number;
  max: number;
  step: number;
}

export interface FnModel {
  type: FitFunctionType;
  domain: [number, number];
  /** evaluate f(x; θ) */
  eval(theta: number[], x: number): number;
  /** ∂f/∂θ at x (same length as θ) */
  grad(theta: number[], x: number): number[];
  /** enforce constraints (σ>0 etc.) */
  clamp(theta: number[]): number[];
  /** deliberately-wrong random starting θ for gradient descent */
  randomInit(): number[];
  /** optional data-aware start that lands in the right basin for hard
   *  (multi-modal) parameters, while leaving amplitudes for GD to learn */
  smartInit?(data: { x: number; y: number }[]): number[];
  /** UI metadata aligned with θ order */
  fields(): ParamField[];
  /** human labels aligned with θ order */
  labels(): string[];
}

// Best frequency k for trig data: for each candidate k, fit A,B by least
// squares (the model is linear in A,B) and keep the k with the smallest error.
function bestTrigFrequency(data: { x: number; y: number }[]): number {
  let bestK = 1, bestSSE = Infinity;
  for (let k = 0.2; k <= 3.0001; k += 0.05) {
    let cc = 0, cs = 0, ss = 0, cy = 0, sy = 0;
    for (const p of data) {
      const c = Math.cos(k * p.x), s = Math.sin(k * p.x);
      cc += c * c; cs += c * s; ss += s * s; cy += c * p.y; sy += s * p.y;
    }
    const det = cc * ss - cs * cs;
    if (Math.abs(det) < 1e-9) continue;
    const A = (cy * ss - sy * cs) / det;
    const B = (sy * cc - cy * cs) / det;
    let sse = 0;
    for (const p of data) {
      const e = p.y - (A * Math.cos(k * p.x) + B * Math.sin(k * p.x));
      sse += e * e;
    }
    if (sse < bestSSE) { bestSSE = sse; bestK = k; }
  }
  return bestK;
}

const SIG_MIN = 0.15;
const rand = (lo: number, hi: number) => lo + Math.random() * (hi - lo);

// ── Default TRUE parameters (what the setup page starts from) ─────────────────
export function defaultTheta(type: FitFunctionType, k = 3): number[] {
  switch (type) {
    case 'line': return [1.5, -0.5];
    case 'trig': return [1.5, 0.8, 1.2];
    case 'gaussian': return [0.5, 1.2];
    case 'gmm': {
      const presets = [
        [1.0, -3, 0.8],
        [0.8, 0, 1.0],
        [1.2, 3, 0.7],
        [0.9, 1.5, 0.6],
      ];
      const out: number[] = [];
      for (let i = 0; i < k; i++) out.push(...presets[i]);
      return out;
    }
  }
}

// ── Model factory ────────────────────────────────────────────────────────────
export function makeModel(type: FitFunctionType, k = 3): FnModel {
  switch (type) {
    case 'line':
      return {
        type, domain: [-5, 5],
        eval: (t, x) => t[0] * x + t[1],
        grad: (_t, x) => [x, 1],
        clamp: t => t,
        randomInit: () => [rand(-3, 3), rand(-3, 3)],
        fields: () => [
          { label: 'a（傾き）', min: -5, max: 5, step: 0.1 },
          { label: 'b（y切片）', min: -5, max: 5, step: 0.1 },
        ],
        labels: () => ['a', 'b'],
      };

    case 'trig':
      return {
        type, domain: [-6, 6],
        eval: (t, x) => t[0] * Math.cos(t[2] * x) + t[1] * Math.sin(t[2] * x),
        grad: (t, x) => {
          const [A, B, kk] = t;
          return [
            Math.cos(kk * x),
            Math.sin(kk * x),
            x * (-A * Math.sin(kk * x) + B * Math.cos(kk * x)),
          ];
        },
        clamp: t => [t[0], t[1], Math.max(t[2], 0.05)],
        randomInit: () => [rand(-2, 2), rand(-2, 2), rand(0.3, 2.5)],
        // Seed the (multi-modal) frequency k in the right basin, but start the
        // amplitudes A,B small so gradient descent still visibly grows them.
        smartInit: data => [rand(-0.6, 0.6), rand(-0.6, 0.6), bestTrigFrequency(data) + rand(-0.08, 0.08)],
        fields: () => [
          { label: 'A（cos係数）', min: -3, max: 3, step: 0.1 },
          { label: 'B（sin係数）', min: -3, max: 3, step: 0.1 },
          { label: 'k（角周波数）', min: 0.1, max: 3, step: 0.05 },
        ],
        labels: () => ['A', 'B', 'k'],
      };

    case 'gaussian':
      return {
        type, domain: [-5, 5],
        eval: (t, x) => {
          const [mu, sg] = t;
          return Math.exp(-((x - mu) ** 2) / (2 * sg * sg));
        },
        grad: (t, x) => {
          const [mu, sg] = t;
          const f = Math.exp(-((x - mu) ** 2) / (2 * sg * sg));
          return [
            f * (x - mu) / (sg * sg),
            f * ((x - mu) ** 2) / (sg * sg * sg),
          ];
        },
        clamp: t => [t[0], Math.max(t[1], SIG_MIN)],
        randomInit: () => [rand(-3, 3), rand(0.5, 2.5)],
        fields: () => [
          { label: 'μ（平均）', min: -4, max: 4, step: 0.1 },
          { label: 'σ（標準偏差）', min: 0.2, max: 3, step: 0.1 },
        ],
        labels: () => ['μ', 'σ'],
      };

    case 'gmm':
      // θ layout grouped per component: [w0,μ0,σ0, w1,μ1,σ1, ...]
      return {
        type, domain: [-7, 7],
        eval: (t, x) => {
          let s = 0;
          for (let i = 0; i < k; i++) {
            const w = t[3 * i], mu = t[3 * i + 1], sg = t[3 * i + 2];
            s += w * Math.exp(-((x - mu) ** 2) / (2 * sg * sg));
          }
          return s;
        },
        grad: (t, x) => {
          const g = new Array(3 * k).fill(0);
          for (let i = 0; i < k; i++) {
            const w = t[3 * i], mu = t[3 * i + 1], sg = t[3 * i + 2];
            const e = Math.exp(-((x - mu) ** 2) / (2 * sg * sg));
            g[3 * i] = e;                                   // ∂/∂w
            g[3 * i + 1] = w * e * (x - mu) / (sg * sg);    // ∂/∂μ
            g[3 * i + 2] = w * e * ((x - mu) ** 2) / (sg * sg * sg); // ∂/∂σ
          }
          return g;
        },
        clamp: t => t.map((v, j) => (j % 3 === 2 ? Math.max(v, SIG_MIN) : v)),
        randomInit: () => {
          // Spread the component means across the domain (+ jitter) so each one
          // covers a region — avoids two components collapsing onto one cluster.
          const out: number[] = [];
          const [lo, hi] = [-7, 7];
          for (let i = 0; i < k; i++) {
            const mu = lo + ((hi - lo) * (i + 0.5)) / k + rand(-1, 1);
            out.push(rand(0.4, 1.2), mu, rand(0.6, 1.4));
          }
          return out;
        },
        fields: () => {
          const f: ParamField[] = [];
          for (let i = 0; i < k; i++) {
            f.push({ label: `成分${i + 1}  w（重み）`, min: 0, max: 2, step: 0.1 });
            f.push({ label: `成分${i + 1}  μ`, min: -6, max: 6, step: 0.1 });
            f.push({ label: `成分${i + 1}  σ`, min: 0.2, max: 3, step: 0.1 });
          }
          return f;
        },
        labels: () => {
          const l: string[] = [];
          for (let i = 0; i < k; i++) l.push(`w${i + 1}`, `μ${i + 1}`, `σ${i + 1}`);
          return l;
        },
      };
  }
}

// ── Sample a curve for plotting ──────────────────────────────────────────────
export function sampleModel(model: FnModel, theta: number[], nPts = 200) {
  const [lo, hi] = model.domain;
  const xs: number[] = [], ys: number[] = [];
  for (let i = 0; i < nPts; i++) {
    const x = lo + ((hi - lo) * i) / (nPts - 1);
    xs.push(x);
    ys.push(model.eval(theta, x));
  }
  return { x: xs, y: ys };
}

import { useMemo, useState, useCallback } from 'react';
import Plot from './Plot';
import { Button, Card, ParamControl, StepHeader } from './ui';
import { darkLayout, plotConfig } from './plotTheme';
import { randn } from '../utils/math';

// ── Overfitting & L2 regularization (ridge polynomial regression) ───────────
// Tiny noisy dataset + polynomial degree 1–9 + weight-decay λ. The ridge fit
// has a closed-form solution, so every slider move re-fits instantly — no
// iteration loop, no divergence, no NaN.

const X_MIN = -1, X_MAX = 1;
const trueFn = (x: number) => Math.sin(3 * x) + 0.3 * x;

const XS = Array.from({ length: 200 }, (_, i) => X_MIN + ((X_MAX - X_MIN) * i) / 199);

interface Pt { x: number; y: number }

/** Solve A·w = b by Gaussian elimination with partial pivoting (A is ≤10×10). */
function solveLinear(A: number[][], b: number[]): number[] {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    [M[col], M[piv]] = [M[piv], M[col]];
    const p = M[col][col];
    if (Math.abs(p) < 1e-12) continue;   // singular direction → leave weight at 0
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = M[r][col] / p;
      for (let c2 = col; c2 <= n; c2++) M[r][c2] -= f * M[col][c2];
    }
  }
  return M.map((row, i) => {
    const p = row[i];
    const w = Math.abs(p) < 1e-12 ? 0 : row[n] / p;
    return isFinite(w) ? w : 0;
  });
}

/** Ridge fit: minimize Σ(poly(x)−y)² + λ·Σ_{j≥1} wⱼ²  (bias not penalized). */
function ridgeFit(pts: Pt[], degree: number, lambda: number): number[] {
  const d = degree + 1;
  const A: number[][] = Array.from({ length: d }, () => Array(d).fill(0));
  const b: number[] = Array(d).fill(0);
  for (const p of pts) {
    const feats: number[] = [];
    let pw = 1;
    for (let j = 0; j < d; j++) { feats.push(pw); pw *= p.x; }
    for (let j = 0; j < d; j++) {
      b[j] += feats[j] * p.y;
      for (let k2 = 0; k2 < d; k2++) A[j][k2] += feats[j] * feats[k2];
    }
  }
  for (let j = 1; j < d; j++) A[j][j] += lambda;   // skip bias (j=0)
  return solveLinear(A, b);
}

function polyEval(w: number[], x: number): number {
  let s = 0, pw = 1;
  for (const wi of w) { s += wi * pw; pw *= x; }
  return s;
}

function rmse(w: number[], pts: Pt[]): number {
  let s = 0;
  for (const p of pts) { const e = polyEval(w, p.x) - p.y; s += e * e; }
  return Math.sqrt(s / pts.length);
}

function makeData(): { train: Pt[]; test: Pt[] } {
  const train: Pt[] = [];
  const nTrain = 12;
  for (let i = 0; i < nTrain; i++) {
    // jittered grid keeps points spread out even with so few samples
    const x = X_MIN + ((X_MAX - X_MIN) * (i + 0.5)) / nTrain + randn() * 0.03;
    train.push({ x, y: trueFn(x) + randn() * 0.3 });
  }
  const test: Pt[] = [];
  for (let i = 0; i < 40; i++) {
    const x = X_MIN + Math.random() * (X_MAX - X_MIN);
    test.push({ x, y: trueFn(x) + randn() * 0.3 });
  }
  return { train, test };
}

export default function OverfittingMode({ onBack }: { onBack: () => void }) {
  const [seed, setSeed] = useState(0);
  const [degree, setDegree] = useState(9);
  const [logLam, setLogLam] = useState(-8);

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const { train, test } = useMemo(() => makeData(), [seed]);
  const lambda = Math.pow(10, logLam);

  const weights = useMemo(() => ridgeFit(train, degree, lambda), [train, degree, lambda]);
  const fitCurve = useMemo(() => XS.map(x => polyEval(weights, x)), [weights]);

  // U-curve: train / test RMSE for every degree at the current λ
  const uCurve = useMemo(() => {
    const degs = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    const trainE: number[] = [], testE: number[] = [];
    for (const dg of degs) {
      const w = ridgeFit(train, dg, lambda);
      trainE.push(rmse(w, train));
      testE.push(rmse(w, test));
    }
    return { degs, trainE, testE };
  }, [train, test, lambda]);

  const regenerate = useCallback(() => setSeed(s => s + 1), []);

  const trainErr = rmse(weights, train);
  const testErr = rmse(weights, test);
  const overfitting = testErr > trainErr * 2 && degree >= 6 && logLam < -4;

  return (
    <div className="min-h-screen px-6 py-6 max-w-6xl mx-auto">
      <StepHeader
        title="過学習と正則化 — モデルの複雑さを体験"
        steps={['タスク選択', '可視化']} current={1} onBack={onBack}
      />

      <div className="grid lg:grid-cols-[300px_1fr] gap-6">
        <div className="space-y-5">
          <Card title="モデルの複雑さ">
            <ParamControl label="多項式の次数" value={degree} min={1} max={9} step={1}
              onChange={v => setDegree(Math.round(v))} />
            <ParamControl label="L2正則化 log₁₀λ" value={logLam} min={-8} max={1} step={0.1}
              onChange={setLogLam} accent="emerald" />
            <p className="text-xs text-slate-500 font-mono mb-2">λ = {lambda.toExponential(1)}</p>
            <div className="rounded-lg bg-indigo-50 border border-indigo-200 p-2.5 text-xs text-indigo-700 leading-relaxed">
              次数を9に上げてλを最小にすると過学習が起きます。そこからλのスライダーを右へ動かすと、重みにペナルティがかかり曲線が滑らかになります。
            </div>
          </Card>

          <Button onClick={regenerate} variant="ghost" className="w-full">🎲 データを再生成</Button>

          <Card title="誤差（RMSE）">
            <div className="space-y-2 text-xs">
              <div className="flex justify-between">
                <span className="text-slate-500">訓練誤差</span>
                <span className="font-mono font-bold text-indigo-600">{trainErr.toFixed(3)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">テスト誤差（未知データ）</span>
                <span className={`font-mono font-bold ${overfitting ? 'text-red-600' : 'text-emerald-600'}`}>
                  {testErr.toFixed(3)}
                </span>
              </div>
            </div>
            {overfitting && (
              <div className="mt-3 rounded-lg bg-red-50 border border-red-200 p-2.5 text-xs text-red-700 leading-relaxed">
                ⚠️ 過学習が起きています！ 訓練データには合っていますが、未知データへの誤差が悪化しています。λを上げてみましょう。
              </div>
            )}
          </Card>
        </div>

        <div className="space-y-5">
          <Card title="フィッティングの様子（●訓練データ ○テストデータ）">
            <div className="h-80">
              <Plot
                data={[
                  { x: test.map(p => p.x), y: test.map(p => p.y), mode: 'markers',
                    marker: { color: 'rgba(148,163,184,0.45)', size: 6, symbol: 'circle-open' },
                    type: 'scatter', name: 'テスト' },
                  { x: XS, y: XS.map(trueFn), mode: 'lines',
                    line: { color: '#cbd5e1', width: 2, dash: 'dash' }, type: 'scatter', name: '真の関数' },
                  { x: XS, y: fitCurve.map(v => Math.max(-3, Math.min(3, v))), mode: 'lines',
                    line: { color: overfitting ? '#dc2626' : '#4f46e5', width: 3 }, type: 'scatter', name: 'モデル' },
                  { x: train.map(p => p.x), y: train.map(p => p.y), mode: 'markers',
                    marker: { color: '#4f46e5', size: 8 }, type: 'scatter', name: '訓練' },
                ] as Plotly.Data[]}
                layout={{
                  ...darkLayout('x', 'y'),
                  height: 310,
                  yaxis: { ...darkLayout().yaxis, range: [-2.2, 2.2] },
                }}
                config={plotConfig}
              />
            </div>
            <p className="text-xs text-slate-500 mt-2">
              曲線が訓練点をすべて通ろうとして波打つのが過学習。λを上げると自然な曲線に戻ります。
            </p>
          </Card>

          <Card title="訓練誤差 vs テスト誤差（次数ごとのU字カーブ）">
            <div className="h-52">
              <Plot
                data={[
                  { x: uCurve.degs, y: uCurve.trainE, mode: 'lines+markers',
                    line: { color: '#4f46e5', width: 2 }, marker: { size: 6 }, type: 'scatter', name: '訓練' },
                  { x: uCurve.degs, y: uCurve.testE, mode: 'lines+markers',
                    line: { color: '#ea580c', width: 2 }, marker: { size: 6 }, type: 'scatter', name: 'テスト' },
                  { x: [degree], y: [testErr], mode: 'markers',
                    marker: { color: '#ea580c', size: 14, symbol: 'circle-open', line: { width: 2 } },
                    type: 'scatter', name: '現在' },
                ] as Plotly.Data[]}
                layout={{ ...darkLayout('多項式の次数', 'RMSE'), height: 195 }}
                config={plotConfig}
              />
            </div>
            <p className="text-xs text-slate-500 mt-2">
              <span className="text-indigo-600 font-medium">— 訓練誤差</span>は複雑にするほど下がり続けますが、
              <span className="text-orange-600 font-medium"> — テスト誤差</span>は途中から悪化（U字）します。
            </p>
          </Card>
        </div>
      </div>
    </div>
  );
}

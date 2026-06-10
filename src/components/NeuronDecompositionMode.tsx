import { useMemo, useRef, useState, useCallback, useEffect } from 'react';
import Plot from './Plot';
import { Button, Card, ParamControl, StepHeader } from './ui';
import { darkLayout, plotConfig, COLORS } from './plotTheme';
import { randn } from '../utils/math';
import { useAnimator } from '../utils/useAnimator';
import { makeModel, defaultTheta } from '../utils/functions';
import type { FitFunctionType } from '../utils/functions';

// ── NN neuron decomposition ──────────────────────────────────────────────────
// A tiny 1-hidden-layer network with Gaussian-basis neurons plus a linear
// skip connection:
//   ŷ(x) = a·x + Σᵢ vᵢ · exp(−(wᵢx + cᵢ)²) + d
// The linear term handles straight-line structure; each hidden neuron adopts
// one local bump, so the per-neuron mini-plots show the division of labour.

const TARGETS: { key: FitFunctionType; label: string }[] = [
  { key: 'line', label: '直線' },
  { key: 'trig', label: '三角関数' },
  { key: 'gaussian', label: 'ガウス分布' },
  { key: 'gmm', label: '混合ガウス' },
];

interface NNParams { w: number[]; c: number[]; v: number[]; d: number; a: number }
interface Snap { p: NNParams; loss: number }

const cloneP = (p: NNParams): NNParams => ({ w: [...p.w], c: [...p.c], v: [...p.v], d: p.d, a: p.a });

function forward(p: NNParams, x: number): number {
  let s = p.a * x + p.d;
  for (let i = 0; i < p.w.length; i++) {
    const u = p.w[i] * x + p.c[i];
    s += p.v[i] * Math.exp(-u * u);
  }
  return s;
}

function neuronOut(p: NNParams, i: number, x: number): number {
  const u = p.w[i] * x + p.c[i];
  return p.v[i] * Math.exp(-u * u);
}

export default function NeuronDecompositionMode({ onBack }: { onBack: () => void }) {
  const [target, setTarget] = useState<FitFunctionType>('gmm');
  const [H, setH] = useState(3);
  const [maxFrames, setMaxFrames] = useState(120);
  const [speed, setSpeed] = useState(80);

  const model = useMemo(() => makeModel(target, 3), [target]);
  const trueTheta = useMemo(() => defaultTheta(target, 3), [target]);
  const [xLo, xHi] = model.domain;

  const XS = useMemo(
    () => Array.from({ length: 160 }, (_, i) => xLo + ((xHi - xLo) * i) / 159),
    [xLo, xHi]
  );

  const data = useMemo(() => {
    const pts: { x: number; y: number }[] = [];
    for (let i = 0; i < 120; i++) {
      const x = xLo + Math.random() * (xHi - xLo);
      pts.push({ x, y: model.eval(trueTheta, x) + randn() * 0.08 });
    }
    pts.sort((a, b) => a.x - b.x);
    return pts;
  }, [model, trueTheta, xLo, xHi]);

  // Adam state lives in refs (single source of truth)
  const pRef = useRef<NNParams | null>(null);
  const mRef = useRef<NNParams | null>(null);   // 1st moments
  const vRef = useRef<NNParams | null>(null);   // 2nd moments
  const tRef = useRef(0);
  const histRef = useRef<Snap[]>([]);
  const [hist, setHist] = useState<Snap[]>([]);
  const [cur, setCur] = useState(0);
  const maxFramesRef = useRef(maxFrames); maxFramesRef.current = maxFrames;

  const lossOf = useCallback((p: NNParams) => {
    let s = 0;
    for (const pt of data) { const e = forward(p, pt.x) - pt.y; s += e * e; }
    return s / data.length;
  }, [data]);

  const resetNN = useCallback(() => {
    // Spread initial neuron centres across the domain (with jitter) so each one
    // starts near a different region — safe, NaN-free initialization.
    const w: number[] = [], c: number[] = [], v: number[] = [];
    for (let i = 0; i < H; i++) {
      const centre = xLo + ((xHi - xLo) * (i + 0.5)) / H + randn() * 0.5;
      const wi = (0.7 + Math.random() * 0.5) * (3 / Math.max(xHi - xLo, 1) * 4);
      w.push(wi);
      c.push(-wi * centre);
      v.push(0.3 + Math.random() * 0.3);
    }
    const p: NNParams = { w, c, v, d: 0, a: 0 };
    pRef.current = p;
    mRef.current = { w: w.map(() => 0), c: c.map(() => 0), v: v.map(() => 0), d: 0, a: 0 };
    vRef.current = { w: w.map(() => 0), c: c.map(() => 0), v: v.map(() => 0), d: 0, a: 0 };
    tRef.current = 0;
    histRef.current = [{ p: cloneP(p), loss: lossOf(p) }];
    setHist([...histRef.current]);
    setCur(0);
  }, [H, lossOf, xLo, xHi]);

  useEffect(() => { resetNN(); }, [resetNN]);

  // One animation frame = a small burst of Adam steps (keeps the loop light)
  const stepOnce = useCallback((): boolean => {
    const p = pRef.current, m = mRef.current, vm = vRef.current;
    if (!p || !m || !vm) return true;
    if (histRef.current.length - 1 >= maxFramesRef.current) return true;

    const lr = 0.05, n = data.length;
    const INNER = 12;

    for (let it = 0; it < INNER; it++) {
      const gw = p.w.map(() => 0), gc = p.c.map(() => 0), gv = p.v.map(() => 0);
      let gd = 0, ga = 0;
      for (const pt of data) {
        const e = (2 / n) * (forward(p, pt.x) - pt.y);
        gd += e;
        ga += e * pt.x;
        for (let i = 0; i < p.w.length; i++) {
          const u = p.w[i] * pt.x + p.c[i];
          const phi = Math.exp(-u * u);
          gv[i] += e * phi;
          const dphi = -2 * u * phi * p.v[i];
          gw[i] += e * dphi * pt.x;
          gc[i] += e * dphi;
        }
      }
      tRef.current += 1;
      const bc1 = 1 - Math.pow(0.9, tRef.current);
      const bc2 = 1 - Math.pow(0.999, tRef.current);
      const upd = (val: number, mm: number, vv: number, g: number): [number, number, number] => {
        const nm = 0.9 * mm + 0.1 * g;
        const nv = 0.999 * vv + 0.001 * g * g;
        const next = val - lr * (nm / bc1) / (Math.sqrt(nv / bc2) + 1e-8);
        return [isFinite(next) ? next : val, nm, nv];
      };
      for (let i = 0; i < p.w.length; i++) {
        [p.w[i], m.w[i], vm.w[i]] = upd(p.w[i], m.w[i], vm.w[i], gw[i]);
        [p.c[i], m.c[i], vm.c[i]] = upd(p.c[i], m.c[i], vm.c[i], gc[i]);
        [p.v[i], m.v[i], vm.v[i]] = upd(p.v[i], m.v[i], vm.v[i], gv[i]);
      }
      [p.d, m.d, vm.d] = upd(p.d, m.d, vm.d, gd);
      [p.a, m.a, vm.a] = upd(p.a, m.a, vm.a, ga);
    }

    histRef.current = [...histRef.current, { p: cloneP(p), loss: lossOf(p) }];
    const step = histRef.current.length - 1;
    setHist([...histRef.current]);
    setCur(step);
    return step >= maxFramesRef.current;
  }, [data, lossOf]);

  const { status, start, pause, step, reset } = useAnimator(stepOnce, resetNN, speed);
  const isRunning = status === 'running';
  const snap = hist[cur];

  const totalCurve = useMemo(
    () => (snap ? XS.map(x => forward(snap.p, x)) : []),
    [snap, XS]
  );
  const trueCurve = useMemo(() => XS.map(x => model.eval(trueTheta, x)), [XS, model, trueTheta]);

  return (
    <div className="min-h-screen px-6 py-6 max-w-6xl mx-auto">
      <StepHeader
        title="ニューラルネット — 隠れ層の役割分担"
        steps={['タスク選択', '可視化']} current={1} onBack={onBack}
      />

      <div className="grid lg:grid-cols-[300px_1fr] gap-6">
        <div className="space-y-5">
          <Card title="フィットする関数">
            <div className="grid grid-cols-2 gap-2">
              {TARGETS.map(t => (
                <button key={t.key} onClick={() => setTarget(t.key)}
                  className={`py-2 rounded-lg text-xs font-medium border transition-all ${
                    target === t.key ? 'bg-indigo-600 border-indigo-600 text-white'
                      : 'bg-white border-slate-300 text-slate-600 hover:border-slate-400'
                  }`}>{t.label}</button>
              ))}
            </div>
          </Card>

          <Card title="ネットワーク設定">
            <label className="block text-xs text-slate-600 mb-1.5">隠れニューロン数 H</label>
            <div className="flex gap-2 mb-3">
              {[1, 2, 3, 4].map(hh => (
                <button key={hh} onClick={() => setH(hh)}
                  className={`flex-1 py-1.5 rounded-lg text-sm font-bold border transition-all ${
                    H === hh ? 'bg-indigo-600 border-indigo-600 text-white'
                      : 'bg-white border-slate-300 text-slate-600 hover:border-slate-400'
                  }`}>{hh}</button>
              ))}
            </div>
            <ParamControl label="最大フレーム数" value={maxFrames} min={30} max={200} step={10}
              onChange={v => setMaxFrames(Math.round(v))} accent="emerald" />
            <ParamControl label="アニメーション速度 (ms)" value={speed} min={30} max={400} step={10}
              onChange={v => setSpeed(Math.round(v))} accent="orange" />
            <div className="rounded-lg bg-indigo-50 border border-indigo-200 p-2.5 text-xs text-indigo-700 leading-relaxed mt-1">
              ŷ(x) = a·x + Σ vᵢ·exp(−(wᵢx+cᵢ)²) + d。線形項 a·x が直線成分を、各ニューロンが局所的な「山」を分担します。Hが足りないと表現しきれない様子も観察できます。
            </div>
          </Card>

          <div className="grid grid-cols-2 gap-2">
            <Button onClick={isRunning ? pause : start} variant={isRunning ? 'amber' : 'primary'}
              disabled={status === 'done'}>
              {isRunning ? '⏸ 一時停止' : status === 'done' ? '✓ 完了' : '▶ 自動実行'}
            </Button>
            <Button onClick={step} variant="ghost" disabled={isRunning || status === 'done'}>⏭ ステップ</Button>
          </div>
          <Button onClick={reset} variant="danger" className="w-full">↺ リセット（初期値を再抽選）</Button>

          <Card title="損失（MSE）">
            <div className="h-32">
              <Plot
                data={[{
                  x: hist.slice(0, cur + 1).map((_, i) => i),
                  y: hist.slice(0, cur + 1).map(h => h.loss),
                  mode: 'lines', line: { color: '#059669', width: 2 },
                  fill: 'tozeroy', fillcolor: 'rgba(5,150,105,0.08)', type: 'scatter',
                }] as Plotly.Data[]}
                layout={{ ...darkLayout('フレーム', 'MSE'), height: 125, margin: { l: 42, r: 8, t: 8, b: 32 } }}
                config={plotConfig}
              />
            </div>
            <p className="text-xs text-slate-500 mt-2 font-mono">
              a = {snap ? snap.p.a.toFixed(2) : '—'}（線形項の係数）
            </p>
          </Card>
        </div>

        <div className="space-y-5">
          <Card title="全体のフィッティング（出力層 = 線形項＋各ニューロンの合算）" right={
            <span className="text-xs font-mono text-indigo-600">Step {cur} / {hist.length - 1}</span>
          }>
            <div className="h-72">
              <Plot
                data={[
                  { x: data.map(p => p.x), y: data.map(p => p.y), mode: 'markers',
                    marker: { color: '#94a3b8', size: 4, opacity: 0.6 }, type: 'scatter' },
                  { x: XS, y: trueCurve, mode: 'lines',
                    line: { color: '#cbd5e1', width: 2, dash: 'dash' }, type: 'scatter' },
                  { x: XS, y: totalCurve, mode: 'lines',
                    line: { color: '#4f46e5', width: 3 }, type: 'scatter' },
                ] as Plotly.Data[]}
                layout={{ ...darkLayout('x', 'y'), height: 280 }}
                config={plotConfig}
              />
            </div>
          </Card>

          <Card title="各ニューロンの出力 vᵢ·φᵢ(x) — 役割分担の様子">
            <div className={`grid gap-3 ${H === 1 ? 'grid-cols-1' : 'grid-cols-2'}`}>
              {snap && Array.from({ length: H }, (_, i) => (
                <div key={i} className="rounded-xl border border-slate-200 overflow-hidden">
                  <div className="px-2.5 py-1 text-[11px] font-semibold border-b border-slate-100"
                    style={{ color: COLORS[i % COLORS.length] }}>
                    ニューロン {i + 1}
                  </div>
                  <div className="h-32">
                    <Plot
                      data={[
                        { x: XS, y: trueCurve, mode: 'lines',
                          line: { color: '#e2e8f0', width: 1.5 }, type: 'scatter' },
                        { x: XS, y: XS.map(x => neuronOut(snap.p, i, x)), mode: 'lines',
                          line: { color: COLORS[i % COLORS.length], width: 2.5 },
                          fill: 'tozeroy',
                          fillcolor: COLORS[i % COLORS.length] + '14',
                          type: 'scatter' },
                      ] as Plotly.Data[]}
                      layout={{ ...darkLayout(), height: 125, margin: { l: 32, r: 6, t: 6, b: 22 } }}
                      config={plotConfig}
                    />
                  </div>
                </div>
              ))}
            </div>
            <p className="text-xs text-slate-500 mt-2 leading-relaxed">
              薄いグレーが目標の関数。学習が進むと、各ニューロンが別々の領域を受け持つように分化していきます。
            </p>
          </Card>

          {(status === 'paused' || status === 'done') && hist.length > 1 && (
            <Card>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-medium text-slate-600">ステップをスクラブ（分化の過程を再生）</span>
                <span className="text-xs font-mono text-indigo-600">Step {cur} / {hist.length - 1}</span>
              </div>
              <input type="range" min={0} max={hist.length - 1} value={cur}
                onChange={e => setCur(Number(e.target.value))}
                className="w-full h-2 appearance-none rounded-full bg-slate-200 accent-indigo-600 cursor-pointer" />
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}

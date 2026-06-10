import { useMemo, useRef, useState, useCallback, useEffect } from 'react';
import Plot from './Plot';
import { Button, Card, ParamControl, StepHeader } from './ui';
import { darkLayout, layout3d, plotConfig } from './plotTheme';
import { randn } from '../utils/math';
import { useAnimator } from '../utils/useAnimator';

// ── 3D Loss Landscape: watch SGD vs Adam roll down the MSE valley ───────────
// Model: y = w·x + b (2 params) → the loss L(w, b) is a 3D surface.

const TRUE_W = 1.8, TRUE_B = -0.7;
const W_MIN = -4, W_MAX = 7, B_MIN = -5.5, B_MAX = 4;

type View = 'both' | 'sgd' | 'adam';

interface Snap {
  sgd: [number, number, number];   // w, b, loss
  adam: [number, number, number];
}

interface AdamMoments { mw: number; mb: number; vw: number; vb: number; t: number }

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

// Purple-tinted surface for the light theme
const SURFACE_SCALE: Array<[number, string]> = [
  [0, '#eef2ff'], [0.35, '#c7d2fe'], [0.7, '#818cf8'], [1, '#4338ca'],
];

export default function LossLandscapeMode({ onBack }: { onBack: () => void }) {
  // Fixed synthetic dataset (regenerated only on mount)
  const data = useMemo(() => {
    const pts: { x: number; y: number }[] = [];
    for (let i = 0; i < 60; i++) {
      const x = -3 + 6 * (i / 59);
      pts.push({ x, y: TRUE_W * x + TRUE_B + randn() * 0.8 });
    }
    return pts;
  }, []);

  const loss = useCallback((w: number, b: number) => {
    let s = 0;
    for (const p of data) { const e = w * p.x + b - p.y; s += e * e; }
    return s / data.length;
  }, [data]);

  // Loss surface grid (computed once per dataset)
  const surface = useMemo(() => {
    const ws: number[] = [], bs: number[] = [];
    for (let w = W_MIN; w <= W_MAX + 1e-9; w += 0.25) ws.push(+w.toFixed(2));
    for (let b = B_MIN; b <= B_MAX + 1e-9; b += 0.25) bs.push(+b.toFixed(2));
    const z = bs.map(b => ws.map(w => loss(w, b)));
    return { ws, bs, z };
  }, [loss]);

  const [sgdLr, setSgdLr] = useState(0.03);
  const [adamLr, setAdamLr] = useState(0.15);
  const [maxIter, setMaxIter] = useState(120);
  const [speed, setSpeed] = useState(80);
  const [view, setView] = useState<View>('both');

  const sgdRef = useRef<[number, number]>([0, 0]);
  const adamRef = useRef<[number, number]>([0, 0]);
  const momentsRef = useRef<AdamMoments>({ mw: 0, mb: 0, vw: 0, vb: 0, t: 0 });
  const histRef = useRef<Snap[]>([]);
  const [hist, setHist] = useState<Snap[]>([]);
  const [cur, setCur] = useState(0);

  const sgdLrRef = useRef(sgdLr); sgdLrRef.current = sgdLr;
  const adamLrRef = useRef(adamLr); adamLrRef.current = adamLr;
  const maxIterRef = useRef(maxIter); maxIterRef.current = maxIter;

  const resetRun = useCallback(() => {
    // Random start high on a slope (same point for both algorithms)
    const corners: [number, number][] = [
      [W_MIN + 0.6, B_MAX - 0.4], [W_MAX - 0.6, B_MAX - 0.4],
      [W_MIN + 0.6, B_MIN + 0.6], [W_MAX - 0.6, B_MIN + 0.6],
    ];
    const [w0, b0] = corners[Math.floor(Math.random() * corners.length)];
    const start: [number, number] = [w0 + randn() * 0.3, b0 + randn() * 0.3];
    sgdRef.current = [...start];
    adamRef.current = [...start];
    momentsRef.current = { mw: 0, mb: 0, vw: 0, vb: 0, t: 0 };
    const l0 = loss(start[0], start[1]);
    histRef.current = [{ sgd: [start[0], start[1], l0], adam: [start[0], start[1], l0] }];
    setHist([...histRef.current]);
    setCur(0);
  }, [loss]);

  useEffect(() => { resetRun(); }, [resetRun]);

  const fullGrad = useCallback((w: number, b: number): [number, number] => {
    let dw = 0, db = 0;
    for (const p of data) {
      const e = w * p.x + b - p.y;
      dw += (2 / data.length) * e * p.x;
      db += (2 / data.length) * e;
    }
    return [dw, db];
  }, [data]);

  const miniGrad = useCallback((w: number, b: number): [number, number] => {
    const B = 6;
    let dw = 0, db = 0;
    for (let i = 0; i < B; i++) {
      const p = data[Math.floor(Math.random() * data.length)];
      const e = w * p.x + b - p.y;
      dw += (2 / B) * e * p.x;
      db += (2 / B) * e;
    }
    return [dw, db];
  }, [data]);

  const stepOnce = useCallback((): boolean => {
    if (histRef.current.length - 1 >= maxIterRef.current) return true;

    // SGD: noisy mini-batch gradient → zigzag descent
    {
      const [w, b] = sgdRef.current;
      const [dw, db] = miniGrad(w, b);
      sgdRef.current = [
        clamp(w - sgdLrRef.current * dw, W_MIN, W_MAX),
        clamp(b - sgdLrRef.current * db, B_MIN, B_MAX),
      ];
    }
    // Adam: full-batch + momentum/normalization → smooth glide
    {
      const [w, b] = adamRef.current;
      const [dw, db] = fullGrad(w, b);
      const m = momentsRef.current;
      m.t += 1;
      m.mw = 0.9 * m.mw + 0.1 * dw; m.mb = 0.9 * m.mb + 0.1 * db;
      m.vw = 0.999 * m.vw + 0.001 * dw * dw; m.vb = 0.999 * m.vb + 0.001 * db * db;
      const bc1 = 1 - Math.pow(0.9, m.t), bc2 = 1 - Math.pow(0.999, m.t);
      adamRef.current = [
        clamp(w - adamLrRef.current * (m.mw / bc1) / (Math.sqrt(m.vw / bc2) + 1e-8), W_MIN, W_MAX),
        clamp(b - adamLrRef.current * (m.mb / bc1) / (Math.sqrt(m.vb / bc2) + 1e-8), B_MIN, B_MAX),
      ];
    }

    const [sw, sb] = sgdRef.current;
    const [aw, ab] = adamRef.current;
    histRef.current = [...histRef.current, {
      sgd: [sw, sb, loss(sw, sb)],
      adam: [aw, ab, loss(aw, ab)],
    }];
    const step = histRef.current.length - 1;
    setHist([...histRef.current]);
    setCur(step);
    return step >= maxIterRef.current;
  }, [miniGrad, fullGrad, loss]);

  const { status, start, pause, step, reset } = useAnimator(stepOnce, resetRun, speed);
  const isRunning = status === 'running';

  const shown = hist.slice(0, cur + 1);
  const zLift = (l: number) => l + 0.4;  // keep the trajectory just above the surface

  const traces3d = useMemo((): Plotly.Data[] => {
    const t: Plotly.Data[] = [{
      type: 'surface',
      x: surface.ws, y: surface.bs, z: surface.z,
      colorscale: SURFACE_SCALE, opacity: 0.92, showscale: false,
      contours: { z: { show: true, usecolormap: true, project: { z: false } } },
    } as unknown as Plotly.Data];

    const path = (key: 'sgd' | 'adam', color: string) => {
      t.push({
        type: 'scatter3d', mode: 'lines+markers',
        x: shown.map(h => h[key][0]), y: shown.map(h => h[key][1]),
        z: shown.map(h => zLift(h[key][2])),
        line: { color, width: 5 },
        marker: { color, size: 2.5 },
      } as unknown as Plotly.Data);
      const last = shown[shown.length - 1][key];
      t.push({
        type: 'scatter3d', mode: 'markers',
        x: [last[0]], y: [last[1]], z: [zLift(last[2])],
        marker: { color, size: 7, symbol: 'circle', line: { color: '#fff', width: 1 } },
      } as unknown as Plotly.Data);
    };
    if (shown.length) {
      if (view !== 'adam') path('sgd', '#ea580c');
      if (view !== 'sgd') path('adam', '#4f46e5');
    }
    return t;
  }, [surface, shown, view]);

  return (
    <div className="min-h-screen px-6 py-6 max-w-6xl mx-auto">
      <StepHeader
        title="3D損失ランドスケープ — SGD vs Adam"
        steps={['タスク選択', '可視化']} current={1} onBack={onBack}
      />

      <div className="grid lg:grid-cols-[300px_1fr] gap-6">
        <div className="space-y-5">
          <Card title="アルゴリズム表示">
            <div className="grid grid-cols-3 gap-2">
              {(['both', 'sgd', 'adam'] as View[]).map(v => (
                <button key={v} onClick={() => setView(v)}
                  className={`py-1.5 rounded-lg text-xs font-bold border transition-all ${
                    view === v ? 'bg-indigo-600 border-indigo-600 text-white'
                      : 'bg-white border-slate-300 text-slate-600 hover:border-slate-400'
                  }`}>
                  {v === 'both' ? '両方' : v.toUpperCase()}
                </button>
              ))}
            </div>
            <div className="mt-3 space-y-1 text-xs">
              <p><span className="inline-block w-3 h-1.5 rounded bg-orange-600 mr-1.5 align-middle" /><span className="text-slate-600">SGD：ミニバッチ勾配 → ジグザグに降下</span></p>
              <p><span className="inline-block w-3 h-1.5 rounded bg-indigo-600 mr-1.5 align-middle" /><span className="text-slate-600">Adam：慣性＋正規化 → 滑らかに谷へ</span></p>
            </div>
          </Card>

          <Card title="ハイパーパラメータ">
            <ParamControl label="SGD 学習率" value={sgdLr} min={0.005} max={0.1} step={0.005} onChange={setSgdLr} accent="orange" />
            <ParamControl label="Adam 学習率" value={adamLr} min={0.02} max={0.5} step={0.01} onChange={setAdamLr} />
            <ParamControl label="最大イテレーション" value={maxIter} min={30} max={200} step={10}
              onChange={v => setMaxIter(Math.round(v))} accent="emerald" />
            <ParamControl label="アニメーション速度 (ms)" value={speed} min={30} max={400} step={10}
              onChange={v => setSpeed(Math.round(v))} accent="orange" />
          </Card>

          <div className="grid grid-cols-2 gap-2">
            <Button onClick={isRunning ? pause : start} variant={isRunning ? 'amber' : 'primary'}
              disabled={status === 'done'}>
              {isRunning ? '⏸ 一時停止' : status === 'done' ? '✓ 完了' : '▶ 自動実行'}
            </Button>
            <Button onClick={step} variant="ghost" disabled={isRunning || status === 'done'}>⏭ ステップ</Button>
          </div>
          <Button onClick={reset} variant="danger" className="w-full">↺ リセット（初期位置を再抽選）</Button>

          <Card title="現在値">
            <div className="space-y-1.5 text-xs font-mono">
              <div className="grid grid-cols-4 gap-1 text-[10px] uppercase tracking-wider text-slate-400 pb-1 border-b border-slate-200">
                <span /><span className="text-right">w</span><span className="text-right">b</span><span className="text-right">Loss</span>
              </div>
              {shown.length > 0 && view !== 'adam' && (
                <div className="grid grid-cols-4 gap-1">
                  <span className="text-orange-600 font-bold">SGD</span>
                  <span className="text-right">{shown[shown.length - 1].sgd[0].toFixed(2)}</span>
                  <span className="text-right">{shown[shown.length - 1].sgd[1].toFixed(2)}</span>
                  <span className="text-right">{shown[shown.length - 1].sgd[2].toFixed(2)}</span>
                </div>
              )}
              {shown.length > 0 && view !== 'sgd' && (
                <div className="grid grid-cols-4 gap-1">
                  <span className="text-indigo-600 font-bold">Adam</span>
                  <span className="text-right">{shown[shown.length - 1].adam[0].toFixed(2)}</span>
                  <span className="text-right">{shown[shown.length - 1].adam[1].toFixed(2)}</span>
                  <span className="text-right">{shown[shown.length - 1].adam[2].toFixed(2)}</span>
                </div>
              )}
              <div className="grid grid-cols-4 gap-1 text-slate-400">
                <span>真値</span>
                <span className="text-right">{TRUE_W.toFixed(2)}</span>
                <span className="text-right">{TRUE_B.toFixed(2)}</span>
                <span className="text-right">—</span>
              </div>
            </div>
          </Card>
        </div>

        <div className="space-y-5">
          <Card title="損失関数の谷を転がり落ちるパラメータ（ドラッグで回転できます）" right={
            <span className="text-xs font-mono text-indigo-600">Step {cur} / {hist.length - 1}</span>
          }>
            <div className="h-[430px]">
              <Plot
                data={traces3d}
                layout={layout3d('w（傾き）', 'b（切片）', 'Loss (MSE)', { height: 420 })}
                config={plotConfig}
              />
            </div>
          </Card>

          <Card title="損失の推移">
            <div className="h-40">
              <Plot
                data={[
                  ...(view !== 'adam' ? [{
                    x: shown.map((_, i) => i), y: shown.map(h => h.sgd[2]),
                    mode: 'lines', line: { color: '#ea580c', width: 2 }, type: 'scatter', name: 'SGD',
                  }] : []),
                  ...(view !== 'sgd' ? [{
                    x: shown.map((_, i) => i), y: shown.map(h => h.adam[2]),
                    mode: 'lines', line: { color: '#4f46e5', width: 2 }, type: 'scatter', name: 'Adam',
                  }] : []),
                ] as Plotly.Data[]}
                layout={{ ...darkLayout('イテレーション', 'Loss'), height: 165 }}
                config={plotConfig}
              />
            </div>
          </Card>

          {(status === 'paused' || status === 'done') && hist.length > 1 && (
            <Card>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-medium text-slate-600">ステップをスクラブ（軌跡を巻き戻し）</span>
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

import { useMemo, useRef, useState, useCallback, useEffect } from 'react';
import Plot from './Plot';
import { Button, Card, ParamControl, StepHeader } from './ui';
import { darkLayout, layout3d, plotConfig } from './plotTheme';
import { randn } from '../utils/math';
import { useAnimator } from '../utils/useAnimator';

// ── 3D Loss Landscape: watch SGD vs Adam roll down different loss surfaces ──
// All landscapes are analytic in 2 parameters (w, b). SGD uses the gradient
// plus noise (simulating mini-batch stochasticity); Adam uses the full
// gradient with momentum + per-parameter normalization.

export interface Landscape {
  key: string;
  label: string;
  desc: string;
  wRange: [number, number];
  bRange: [number, number];
  loss(w: number, b: number): number;
  grad(w: number, b: number): [number, number];
  start(): [number, number];
  sgdLr: number;
  adamLr: number;
  noise: number;     // gradient-noise scale for SGD
  maxIter: number;
  marks: { w: number; b: number; label: string }[];
}

const jitter = (v: number, s: number) => v + randn() * s;

export const LANDSCAPES: Landscape[] = [
  {
    key: 'convex',
    label: '凸（線形回帰）',
    desc: '線形回帰 y=wx+b のMSE。谷がひとつだけの凸関数なので、どちらのアルゴリズムも必ず最適解に到達します。',
    wRange: [-4, 7], bRange: [-5.5, 4],
    // E[(Δw·x + Δb − ε)²] for x~U(−3,3): 3Δw² + Δb² + σ²
    loss: (w, b) => 3 * (w - 1.8) ** 2 + (b + 0.7) ** 2 + 0.6,
    grad: (w, b) => [6 * (w - 1.8), 2 * (b + 0.7)],
    start: () => {
      const corners: [number, number][] = [[-3.4, 3.4], [6.4, 3.4], [-3.4, -4.9], [6.4, -4.9]];
      const [w, b] = corners[Math.floor(Math.random() * corners.length)];
      return [jitter(w, 0.3), jitter(b, 0.3)];
    },
    sgdLr: 0.03, adamLr: 0.15, noise: 1.5, maxIter: 120,
    marks: [{ w: 1.8, b: -0.7, label: '★ 最適解' }],
  },
  {
    key: 'localmin',
    label: '局所解あり',
    desc: '大域解へ向かう斜面の途中に小さな「落とし穴（局所解）」がある非凸関数。慣性を持たないSGDは穴に捕まり、Adamはモーメンタムで通り抜けて大域解に到達します。',
    wRange: [-5, 5.5], bRange: [-4.5, 5],
    // Broad bowl centred on the GLOBAL minimum, a deep pit at its centre, and a
    // small shallow pit on the slope along the descent path (the trap).
    loss: (w, b) => {
      const d2g = (w - 2.4) ** 2 + (b + 1.4) ** 2;
      const d2p = (w + 2.0) ** 2 + (b - 2.0) ** 2;
      return 2.6 + 0.03 * d2g - 2.0 * Math.exp(-d2g / 2.0) - 0.55 * Math.exp(-d2p / 0.4);
    },
    grad: (w, b) => {
      const d2g = (w - 2.4) ** 2 + (b + 1.4) ** 2;
      const d2p = (w + 2.0) ** 2 + (b - 2.0) ** 2;
      const eg = Math.exp(-d2g / 2.0);
      const ep = Math.exp(-d2p / 0.4);
      return [
        0.06 * (w - 2.4) + 2.0 * eg * (w - 2.4) + (0.55 * 2 / 0.4) * ep * (w + 2.0),
        0.06 * (b + 1.4) + 2.0 * eg * (b + 1.4) + (0.55 * 2 / 0.4) * ep * (b - 2.0),
      ];
    },
    start: () => [jitter(-3.6, 0.15), jitter(3.2, 0.15)],
    sgdLr: 0.5, adamLr: 0.3, noise: 0.2, maxIter: 150,
    marks: [
      { w: 2.4, b: -1.4, label: '★ 大域解' },
      { w: -2.0, b: 2.0, label: '▲ 局所解' },
    ],
  },
  {
    key: 'valley',
    label: '細長い谷',
    desc: 'Rosenbrock型の曲がった細い谷。SGDは急な壁にジグザグと跳ね返されながら進み、Adamは勾配を正規化して谷底を滑らかに進みます。',
    wRange: [-2, 2], bRange: [-1, 3.5],
    loss: (w, b) => 0.5 * (1 - w) ** 2 + 3 * (b - w * w) ** 2 + 0.05,
    grad: (w, b) => [
      (w - 1) - 12 * w * (b - w * w),
      6 * (b - w * w),
    ],
    start: () => [jitter(-1.6, 0.1), jitter(3.1, 0.1)],
    sgdLr: 0.02, adamLr: 0.08, noise: 0.6, maxIter: 150,
    marks: [{ w: 1, b: 1, label: '★ 最適解' }],
  },
];

type View = 'both' | 'sgd' | 'adam';

interface Snap {
  sgd: [number, number, number];   // w, b, loss
  adam: [number, number, number];
}

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

// Purple-tinted surface for the light theme
const SURFACE_SCALE: Array<[number, string]> = [
  [0, '#eef2ff'], [0.35, '#c7d2fe'], [0.7, '#818cf8'], [1, '#4338ca'],
];

export default function LossLandscapeMode({ onBack }: { onBack: () => void }) {
  const [lsKey, setLsKey] = useState('convex');
  const ls = useMemo(() => LANDSCAPES.find(l => l.key === lsKey)!, [lsKey]);

  const [sgdLr, setSgdLr] = useState(ls.sgdLr);
  const [adamLr, setAdamLr] = useState(ls.adamLr);
  const [maxIter, setMaxIter] = useState(ls.maxIter);
  const [speed, setSpeed] = useState(80);
  const [view, setView] = useState<View>('both');

  // Loss surface grid + z scale info
  const surface = useMemo(() => {
    const [wLo, wHi] = ls.wRange, [bLo, bHi] = ls.bRange;
    const ws: number[] = [], bs: number[] = [];
    const NW = 45, NB = 45;
    for (let i = 0; i < NW; i++) ws.push(wLo + ((wHi - wLo) * i) / (NW - 1));
    for (let i = 0; i < NB; i++) bs.push(bLo + ((bHi - bLo) * i) / (NB - 1));
    const z = bs.map(b => ws.map(w => ls.loss(w, b)));
    let zMin = Infinity, zMax = -Infinity;
    for (const row of z) for (const v of row) { if (v < zMin) zMin = v; if (v > zMax) zMax = v; }
    return { ws, bs, z, lift: (zMax - zMin) * 0.03 };
  }, [ls]);

  const sgdRef = useRef<[number, number]>([0, 0]);
  const adamRef = useRef<[number, number]>([0, 0]);
  const momentsRef = useRef({ mw: 0, mb: 0, vw: 0, vb: 0, t: 0 });
  const histRef = useRef<Snap[]>([]);
  const [hist, setHist] = useState<Snap[]>([]);
  const [cur, setCur] = useState(0);

  const sgdLrRef = useRef(sgdLr); sgdLrRef.current = sgdLr;
  const adamLrRef = useRef(adamLr); adamLrRef.current = adamLr;
  const maxIterRef = useRef(maxIter); maxIterRef.current = maxIter;
  const lsRef = useRef(ls); lsRef.current = ls;

  const resetRun = useCallback(() => {
    const start = lsRef.current.start();
    sgdRef.current = [...start] as [number, number];
    adamRef.current = [...start] as [number, number];
    momentsRef.current = { mw: 0, mb: 0, vw: 0, vb: 0, t: 0 };
    const l0 = lsRef.current.loss(start[0], start[1]);
    histRef.current = [{ sgd: [start[0], start[1], l0], adam: [start[0], start[1], l0] }];
    setHist([...histRef.current]);
    setCur(0);
  }, []);

  // Switching landscapes resets the run and restores its recommended settings
  useEffect(() => {
    setSgdLr(ls.sgdLr);
    setAdamLr(ls.adamLr);
    setMaxIter(ls.maxIter);
    resetRun();
  }, [ls, resetRun]);

  const stepOnce = useCallback((): boolean => {
    if (histRef.current.length - 1 >= maxIterRef.current) return true;
    const L = lsRef.current;
    const [wLo, wHi] = L.wRange, [bLo, bHi] = L.bRange;

    // SGD: gradient + noise → zigzag, no momentum (can get stuck in local minima)
    {
      const [w, b] = sgdRef.current;
      const [dw, db] = L.grad(w, b);
      sgdRef.current = [
        clamp(w - sgdLrRef.current * (dw + L.noise * randn()), wLo, wHi),
        clamp(b - sgdLrRef.current * (db + L.noise * randn()), bLo, bHi),
      ];
    }
    // Adam: momentum + normalization → keeps speed across plateaus & shallow pits
    {
      const [w, b] = adamRef.current;
      const [dw, db] = L.grad(w, b);
      const m = momentsRef.current;
      m.t += 1;
      m.mw = 0.9 * m.mw + 0.1 * dw; m.mb = 0.9 * m.mb + 0.1 * db;
      m.vw = 0.999 * m.vw + 0.001 * dw * dw; m.vb = 0.999 * m.vb + 0.001 * db * db;
      const bc1 = 1 - Math.pow(0.9, m.t), bc2 = 1 - Math.pow(0.999, m.t);
      adamRef.current = [
        clamp(w - adamLrRef.current * (m.mw / bc1) / (Math.sqrt(m.vw / bc2) + 1e-8), wLo, wHi),
        clamp(b - adamLrRef.current * (m.mb / bc1) / (Math.sqrt(m.vb / bc2) + 1e-8), bLo, bHi),
      ];
    }

    const [sw, sb] = sgdRef.current;
    const [aw, ab] = adamRef.current;
    histRef.current = [...histRef.current, {
      sgd: [sw, sb, L.loss(sw, sb)],
      adam: [aw, ab, L.loss(aw, ab)],
    }];
    const step = histRef.current.length - 1;
    setHist([...histRef.current]);
    setCur(step);
    return step >= maxIterRef.current;
  }, []);

  const { status, start, pause, step, reset } = useAnimator(stepOnce, resetRun, speed);
  const isRunning = status === 'running';

  const shown = hist.slice(0, cur + 1);
  const zLift = (l: number) => l + surface.lift;

  const traces3d = useMemo((): Plotly.Data[] => {
    const t: Plotly.Data[] = [{
      type: 'surface',
      x: surface.ws, y: surface.bs, z: surface.z,
      colorscale: SURFACE_SCALE, opacity: 0.92, showscale: false,
      contours: { z: { show: true, usecolormap: true, project: { z: false } } },
    } as unknown as Plotly.Data];

    // Minimum markers (★ global / ▲ local)
    t.push({
      type: 'scatter3d', mode: 'text',
      x: ls.marks.map(m => m.w), y: ls.marks.map(m => m.b),
      z: ls.marks.map(m => ls.loss(m.w, m.b) + surface.lift * 3),
      text: ls.marks.map(m => m.label),
      textfont: { size: 12, color: '#334155' },
    } as unknown as Plotly.Data);

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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [surface, shown, view, ls]);

  return (
    <div className="min-h-screen px-6 py-6 max-w-6xl mx-auto">
      <StepHeader
        title="3D損失ランドスケープ — SGD vs Adam"
        steps={['タスク選択', '可視化']} current={1} onBack={onBack}
      />

      <div className="grid lg:grid-cols-[300px_1fr] gap-6">
        <div className="space-y-5">
          <Card title="損失関数の形">
            <div className="space-y-2">
              {LANDSCAPES.map(l => (
                <button key={l.key} onClick={() => setLsKey(l.key)}
                  className={`w-full py-2 px-3 rounded-lg text-xs font-medium text-left transition-all border ${
                    lsKey === l.key ? 'bg-indigo-600 border-indigo-600 text-white'
                      : 'bg-white border-slate-300 text-slate-600 hover:border-slate-400'
                  }`}>
                  {l.label}
                </button>
              ))}
            </div>
            <p className="text-xs text-slate-500 mt-3 leading-relaxed">{ls.desc}</p>
          </Card>

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
              <p><span className="inline-block w-3 h-1.5 rounded bg-orange-600 mr-1.5 align-middle" /><span className="text-slate-600">SGD：ノイズ入り勾配・慣性なし</span></p>
              <p><span className="inline-block w-3 h-1.5 rounded bg-indigo-600 mr-1.5 align-middle" /><span className="text-slate-600">Adam：慣性＋勾配の正規化</span></p>
            </div>
          </Card>

          <Card title="ハイパーパラメータ">
            <ParamControl label="SGD 学習率" value={sgdLr} min={0.005} max={1} step={0.005} onChange={setSgdLr} accent="orange" />
            <ParamControl label="Adam 学習率" value={adamLr} min={0.02} max={0.6} step={0.01} onChange={setAdamLr} />
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
                layout={layout3d('w', 'b', 'Loss', { height: 420 })}
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
            {lsKey === 'localmin' && (
              <p className="text-xs text-slate-500 mt-2">
                SGD（橙）が高い損失で頭打ちになり、Adam（藍）だけが低い損失まで下がれば、SGDは局所解・Adamは大域解に到達しています。
              </p>
            )}
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

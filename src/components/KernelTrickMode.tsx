import { useMemo, useRef, useState, useCallback, useEffect } from 'react';
import Plot from './Plot';
import { Button, Card, StepHeader } from './ui';
import { darkLayout, layout3d, plotConfig } from './plotTheme';
import { randn } from '../utils/math';

// ── SVM kernel trick (3D lift) ───────────────────────────────────────────────
// Donut data is NOT linearly separable in 2D. The radial feature map
//   φ(x, y) = [x, y, x² + y²]
// lifts it into 3D where a simple horizontal plane separates the classes.

interface Pt { x: number; y: number }

function makeDonut(): { inner: Pt[]; outer: Pt[] } {
  const inner: Pt[] = [], outer: Pt[] = [];
  for (let i = 0; i < 70; i++) {
    const th = Math.random() * 2 * Math.PI;
    const r = Math.abs(0.7 + randn() * 0.25);
    inner.push({ x: r * Math.cos(th), y: r * Math.sin(th) });
  }
  for (let i = 0; i < 90; i++) {
    const th = Math.random() * 2 * Math.PI;
    const r = 2.4 + randn() * 0.25;
    outer.push({ x: r * Math.cos(th), y: r * Math.sin(th) });
  }
  return { inner, outer };
}

const zOf = (p: Pt) => p.x * p.x + p.y * p.y;
const easeOut = (t: number) => 1 - Math.pow(1 - t, 3);

export default function KernelTrickMode({ onBack }: { onBack: () => void }) {
  const [seed, setSeed] = useState(0);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const { inner, outer } = useMemo(() => makeDonut(), [seed]);

  // Separating height: midway between the two classes in feature space
  const planeZ = useMemo(() => {
    const maxInner = Math.max(...inner.map(zOf));
    const minOuter = Math.min(...outer.map(zOf));
    return (maxInner + minOuter) / 2;
  }, [inner, outer]);

  const [t, setT] = useState(0);            // lift progress 0 → 1
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopAnim = useCallback(() => {
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
  }, []);

  const lift = useCallback(() => {
    stopAnim();
    let raw = 0;
    timerRef.current = setInterval(() => {
      raw += 0.025;
      if (raw >= 1) { setT(1); stopAnim(); return; }
      setT(easeOut(raw));
    }, 25);
  }, [stopAnim]);

  const resetLift = useCallback(() => { stopAnim(); setT(0); }, [stopAnim]);
  const regenerate = useCallback(() => { stopAnim(); setT(0); setSeed(s => s + 1); }, [stopAnim]);

  useEffect(() => () => stopAnim(), [stopAnim]);

  const lifted = t > 0.97;
  const circleR = Math.sqrt(planeZ);
  const circle = useMemo(() => {
    const xs: number[] = [], ys: number[] = [];
    for (let i = 0; i <= 80; i++) {
      const th = (2 * Math.PI * i) / 80;
      xs.push(circleR * Math.cos(th));
      ys.push(circleR * Math.sin(th));
    }
    return { xs, ys };
  }, [circleR]);

  // ── 3D traces ──────────────────────────────────────────────────────────────
  const traces3d = useMemo((): Plotly.Data[] => {
    const tr: Plotly.Data[] = [
      {
        type: 'scatter3d', mode: 'markers',
        x: inner.map(p => p.x), y: inner.map(p => p.y),
        z: inner.map(p => t * zOf(p)),
        marker: { color: '#4f46e5', size: 3.5, opacity: 0.85 },
      } as unknown as Plotly.Data,
      {
        type: 'scatter3d', mode: 'markers',
        x: outer.map(p => p.x), y: outer.map(p => p.y),
        z: outer.map(p => t * zOf(p)),
        marker: { color: '#ea580c', size: 3.5, opacity: 0.85 },
      } as unknown as Plotly.Data,
    ];
    if (t > 0.25) {
      // Hyperplane z = planeZ rises together with the data
      const g = [-3.3, 3.3];
      tr.push({
        type: 'surface',
        x: g, y: g,
        z: [[t * planeZ, t * planeZ], [t * planeZ, t * planeZ]],
        colorscale: [[0, '#8b5cf6'], [1, '#8b5cf6']],
        opacity: Math.min(0.35, (t - 0.25) * 0.6),
        showscale: false,
      } as unknown as Plotly.Data);
    }
    return tr;
  }, [inner, outer, t, planeZ]);

  return (
    <div className="min-h-screen px-6 py-6 max-w-6xl mx-auto">
      <StepHeader
        title="SVMとカーネル・トリック — 高次元への写像"
        steps={['タスク選択', '可視化']} current={1} onBack={onBack}
      />

      <div className="grid lg:grid-cols-[300px_1fr] gap-6">
        <div className="space-y-5">
          <Card title="カーネル写像">
            <div className="rounded-lg bg-indigo-50 border border-indigo-200 p-2.5 text-xs text-indigo-700 leading-relaxed font-mono">
              φ(x, y) = [x, y, x² + y²]
            </div>
            <p className="text-xs text-slate-500 mt-2 leading-relaxed">
              2次元では直線で分離できないドーナツ型データも、半径方向の特徴 x²+y² を3次元目に加えると、
              <strong className="text-slate-700">1枚の平面（ハイパープレーン）</strong>で分離できます。
            </p>
          </Card>

          <Button onClick={lift} className="w-full" disabled={t > 0 && t < 1}>
            {t === 0 ? '✨ カーネルトリック実行' : lifted ? '✓ 写像完了' : '写像中…'}
          </Button>
          <div className="grid grid-cols-2 gap-2">
            <Button onClick={resetLift} variant="ghost">↺ 2Dに戻す</Button>
            <Button onClick={regenerate} variant="danger">🎲 データ再生成</Button>
          </div>

          <Card title="状態">
            <div className="space-y-2 text-xs">
              <div className="flex justify-between">
                <span className="text-slate-500">写像の進行</span>
                <span className="font-mono font-bold text-indigo-600">{Math.round(t * 100)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">分離平面の高さ z</span>
                <span className="font-mono font-bold text-violet-600">{planeZ.toFixed(2)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">2Dでの決定境界</span>
                <span className="font-mono font-bold text-emerald-600">
                  {lifted ? `半径 ${circleR.toFixed(2)} の円` : '—'}
                </span>
              </div>
            </div>
            {lifted && (
              <div className="mt-3 rounded-lg bg-emerald-50 border border-emerald-200 p-2.5 text-xs text-emerald-700 leading-relaxed">
                ✓ 3次元では平面 z = {planeZ.toFixed(2)} が2クラスを分離。これを2次元に戻すと「円」の境界になります。
              </div>
            )}
          </Card>
        </div>

        <div className="space-y-5">
          <Card title="3次元の特徴空間（ドラッグで回転できます）" right={
            <span className="text-xs font-mono text-violet-600">z = {(t * 1).toFixed(2)} × (x²+y²)</span>
          }>
            <div className="h-[400px]">
              <Plot
                data={traces3d}
                layout={layout3d('x', 'y', 'x² + y²', { height: 390 })}
                config={plotConfig}
              />
            </div>
          </Card>

          <Card title="元の2次元平面">
            <div className="h-72">
              <Plot
                data={[
                  { x: inner.map(p => p.x), y: inner.map(p => p.y), mode: 'markers',
                    marker: { color: '#4f46e5', size: 6, opacity: 0.8 }, type: 'scatter', name: 'クラスA' },
                  { x: outer.map(p => p.x), y: outer.map(p => p.y), mode: 'markers',
                    marker: { color: '#ea580c', size: 6, opacity: 0.8 }, type: 'scatter', name: 'クラスB' },
                  ...(lifted ? [{
                    x: circle.xs, y: circle.ys, mode: 'lines',
                    line: { color: '#059669', width: 3, dash: 'dash' }, type: 'scatter', name: '決定境界',
                  }] : []),
                ] as Plotly.Data[]}
                layout={{
                  ...darkLayout('x', 'y'),
                  height: 280,
                  yaxis: { ...darkLayout().yaxis, scaleanchor: 'x' },
                  // eslint-disable-next-line @typescript-eslint/no-explicit-any
                } as any}
                config={plotConfig}
              />
            </div>
            <p className="text-xs text-slate-500 mt-2">
              {lifted
                ? '緑の破線が、3次元の分離平面を2次元に「降ろした」決定境界（円）です。'
                : 'この配置では、どんな直線を引いても2クラスを分離できません。'}
            </p>
          </Card>
        </div>
      </div>
    </div>
  );
}

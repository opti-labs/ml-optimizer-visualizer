import { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import Plot from '../components/Plot';
import { Button, Card, ParamControl, StepHeader } from '../components/ui';
import { darkLayout, plotConfig, COLORS } from '../components/plotTheme';
import type { DataPoint } from '../utils/dataGen';
import type { KMeansParams, KMeansSnapshot } from '../utils/kmeans';
import { initKMeans, kmeansInitialSnapshot, kmeansStep } from '../utils/kmeans';
import { useAnimator } from '../utils/useAnimator';

export default function ClusterRunPage({
  k, data, onBack,
}: { k: number; data: DataPoint[]; onBack: () => void }) {
  const [maxIter, setMaxIter] = useState(40);
  const [speed, setSpeed] = useState(350);

  const paramRef = useRef<KMeansParams | null>(null);
  const histRef = useRef<KMeansSnapshot[]>([]);
  const [hist, setHist] = useState<KMeansSnapshot[]>([]);
  const [cur, setCur] = useState(0);

  const maxIterRef = useRef(maxIter); maxIterRef.current = maxIter;

  const resetKM = useCallback(() => {
    const p = initKMeans(k, data);
    paramRef.current = p;
    histRef.current = [kmeansInitialSnapshot(data, p)];
    setHist([...histRef.current]);
    setCur(0);
  }, [k, data]);

  useEffect(() => { resetKM(); }, [resetKM]);

  const stepOnce = useCallback((): boolean => {
    if (!paramRef.current) return true;
    if (histRef.current.length - 1 >= maxIterRef.current) return true;
    const { params, snapshot } = kmeansStep(data, paramRef.current);
    paramRef.current = params;
    histRef.current = [...histRef.current, snapshot];
    const step = histRef.current.length - 1;
    setHist([...histRef.current]);
    setCur(step);
    return snapshot.movement < 1e-6 || step >= maxIterRef.current;
  }, [data]);

  const { status, start, pause, step, reset } = useAnimator(stepOnce, resetKM, speed);
  const isRunning = status === 'running';
  const snap = hist[cur];

  const mainData = useMemo((): Plotly.Data[] => {
    if (!snap) return [];
    const pointsTrace = {
      x: data.map(p => p.x), y: data.map(p => p.y), mode: 'markers',
      marker: { color: data.map((_, i) => COLORS[snap.assignments[i] % COLORS.length]), size: 6, opacity: 0.5 },
      type: 'scatter',
    };
    const trails = [];
    for (let c = 0; c < k; c++) {
      trails.push({
        x: hist.slice(0, cur + 1).map(h => h.centroids[c][0]),
        y: hist.slice(0, cur + 1).map(h => h.centroids[c][1]),
        mode: 'lines+markers',
        line: { color: COLORS[c % COLORS.length], width: 1, dash: 'dot' },
        marker: { color: COLORS[c % COLORS.length], size: 4, opacity: 0.6 },
        type: 'scatter',
      });
    }
    const centroidTrace = {
      x: snap.centroids.map(c => c[0]), y: snap.centroids.map(c => c[1]), mode: 'markers',
      marker: {
        color: snap.centroids.map((_, i) => COLORS[i % COLORS.length]),
        size: 20, symbol: 'x', line: { color: '#0a0c14', width: 2 },
      },
      type: 'scatter',
    };
    return [pointsTrace, ...trails, centroidTrace] as Plotly.Data[];
  }, [data, snap, hist, cur, k]);

  const converged = snap && snap.movement < 1e-4 && cur > 0;

  return (
    <div className="min-h-screen px-6 py-6 max-w-6xl mx-auto">
      <StepHeader
        title="クラスタリング — K平均法"
        steps={['タスク選択', 'データ設定', 'K平均法']}
        current={2}
        onBack={onBack}
      />

      <div className="grid lg:grid-cols-[300px_1fr] gap-6">
        <div className="space-y-5">
          <Card title="設定">
            <ParamControl label="最大イテレーション" value={maxIter} min={5} max={100} step={1}
              onChange={v => setMaxIter(Math.round(v))} accent="emerald" />
            <ParamControl label="アニメーション速度 (ms)" value={speed} min={50} max={1000} step={50}
              onChange={v => setSpeed(Math.round(v))} accent="orange" />
            <div className="rounded-lg bg-emerald-950/40 border border-emerald-800/40 p-2.5 text-xs text-emerald-300 leading-relaxed mt-1">
              ✕ が重心です。「割り当て→重心を平均へ移動」を繰り返し、重心が動かなくなれば収束です。
            </div>
            <div className="rounded-lg bg-amber-950/40 border border-amber-800/40 p-2.5 text-xs text-amber-300 leading-relaxed mt-2">
              💡 収束が早いので、「ステップ」ボタンで1反復ずつ更新するのがおすすめです。
            </div>
          </Card>

          <div className="grid grid-cols-2 gap-2">
            <Button onClick={isRunning ? pause : start} variant={isRunning ? 'amber' : 'primary'}
              disabled={status === 'done'}>
              {isRunning ? '⏸ 一時停止' : status === 'done' ? '✓ 完了' : '▶ 自動実行'}
            </Button>
            <Button onClick={step} variant="ghost" disabled={isRunning || status === 'done'}>
              ⏭ ステップ
            </Button>
          </div>
          <Button onClick={reset} variant="danger" className="w-full">↺ リセット（初期重心を再抽選）</Button>

          <Card title="指標">
            <div className="space-y-2">
              <Metric label="イナーシャ" value={snap ? snap.inertia.toFixed(1) : '—'} />
              <Metric label="重心移動量" value={snap ? snap.movement.toFixed(3) : '—'} />
              <Metric label="状態" value={converged ? '✓ 収束' : isRunning ? '更新中' : '—'} />
            </div>
          </Card>
        </div>

        <div className="space-y-5">
          <Card title="重心(✕)の移動とクラスター割り当て" right={
            <span className="text-xs font-mono text-indigo-400">Step {cur} / {hist.length - 1}</span>
          }>
            <div className="h-96">
              <Plot data={mainData} layout={{ ...darkLayout('x₁', 'x₂'), height: 380 }} config={plotConfig} />
            </div>
          </Card>

          <Card title="イナーシャ（クラスター内平方和）の推移">
            <div className="h-40">
              <Plot
                data={[{
                  x: hist.slice(0, cur + 1).map((_, i) => i),
                  y: hist.slice(0, cur + 1).map(h => h.inertia),
                  mode: 'lines+markers', line: { color: '#34d399', width: 2 },
                  marker: { color: '#34d399', size: 5 },
                  fill: 'tozeroy', fillcolor: 'rgba(52,211,153,0.08)', type: 'scatter',
                }] as Plotly.Data[]}
                layout={{ ...darkLayout('イテレーション', 'イナーシャ'), height: 165 }}
                config={plotConfig}
              />
            </div>
          </Card>

          {(status === 'paused' || status === 'done') && hist.length > 1 && (
            <Card>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-medium text-slate-300">ステップをスクラブ（重心の移動を再生）</span>
                <span className="text-xs font-mono text-indigo-400">Step {cur} / {hist.length - 1}</span>
              </div>
              <input type="range" min={0} max={hist.length - 1} value={cur}
                onChange={e => setCur(Number(e.target.value))}
                className="w-full h-2 appearance-none rounded-full bg-slate-600 accent-emerald-500 cursor-pointer" />
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between text-xs">
      <span className="text-slate-400">{label}</span>
      <span className="font-mono font-bold text-emerald-300">{value}</span>
    </div>
  );
}

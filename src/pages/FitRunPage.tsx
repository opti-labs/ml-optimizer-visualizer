import { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import Plot from '../components/Plot';
import { Button, Card, ParamControl, StepHeader } from '../components/ui';
import { darkLayout, plotConfig } from '../components/plotTheme';
import type { DataPoint } from '../utils/dataGen';
import { makeModel, sampleModel } from '../utils/functions';
import type { FitState } from '../utils/fitGD';
import { initFitState, fitLoss, fitGDStep } from '../utils/fitGD';
import { useAnimator } from '../utils/useAnimator';
import type { FitConfig } from './FitSetupPage';

interface Snapshot { theta: number[]; loss: number; }

export default function FitRunPage({
  config, data, onBack,
}: { config: FitConfig; data: DataPoint[]; onBack: () => void }) {
  const model = useMemo(() => makeModel(config.type, config.k), [config.type, config.k]);

  const [lr, setLr] = useState(0.05);
  const [maxIter, setMaxIter] = useState(120);
  const [speed, setSpeed] = useState(120);

  // refs are the source of truth; state mirrors drive renders
  const stateRef = useRef<FitState | null>(null);
  const histRef = useRef<Snapshot[]>([]);
  const [hist, setHist] = useState<Snapshot[]>([]);
  const [cur, setCur] = useState(0);
  const [diverged, setDiverged] = useState(false);

  const lrRef = useRef(lr); lrRef.current = lr;
  const maxIterRef = useRef(maxIter); maxIterRef.current = maxIter;

  const resetFit = useCallback(() => {
    const st = initFitState(model, data);
    stateRef.current = st;
    histRef.current = [{ theta: st.theta, loss: fitLoss(data, model, st.theta) }];
    setHist([...histRef.current]);
    setCur(0);
    setDiverged(false);
  }, [model, data]);

  // (re)initialize whenever the model or dataset changes
  useEffect(() => { resetFit(); }, [resetFit]);

  const stepOnce = useCallback((): boolean => {
    if (!stateRef.current) return true;
    if (histRef.current.length - 1 >= maxIterRef.current) return true;

    const { state, loss } = fitGDStep(data, model, stateRef.current, lrRef.current);
    stateRef.current = state;
    histRef.current = [...histRef.current, { theta: state.theta, loss }];
    const step = histRef.current.length - 1;
    setHist([...histRef.current]);
    setCur(step);

    const bad = !isFinite(loss) || loss > 1e8 || state.theta.some(v => !isFinite(v));
    if (bad) setDiverged(true);
    return bad || step >= maxIterRef.current;
  }, [data, model]);

  const { status, start, pause, step, reset } = useAnimator(stepOnce, resetFit, speed);

  const isRunning = status === 'running';
  const snap = hist[cur];
  const trueCurve = useMemo(() => sampleModel(model, config.theta, 240), [model, config.theta]);
  const fitCurve = useMemo(
    () => (snap ? sampleModel(model, snap.theta, 240) : { x: [], y: [] }),
    [model, snap]
  );

  const labels = model.labels();

  return (
    <div className="min-h-screen px-6 py-6 max-w-6xl mx-auto">
      <StepHeader
        title="関数フィット — 勾配降下法 (Adam)"
        steps={['タスク選択', 'データ設定', '勾配降下法']}
        current={2}
        onBack={onBack}
      />

      <div className="grid lg:grid-cols-[300px_1fr] gap-6">
        {/* Controls */}
        <div className="space-y-5">
          <Card title="ハイパーパラメータ">
            <ParamControl label="学習率 (lr)" value={lr} min={0.001} max={1} step={0.001}
              onChange={setLr} />
            <ParamControl label="最大イテレーション" value={maxIter} min={20} max={400} step={10}
              onChange={v => setMaxIter(Math.round(v))} accent="emerald" />
            <ParamControl label="アニメーション速度 (ms)" value={speed} min={30} max={600} step={10}
              onChange={v => setSpeed(Math.round(v))} accent="orange" />
            {lr > 0.4 && (
              <div className="rounded-lg bg-amber-50 border border-amber-200 p-2 text-xs text-amber-700">
                ⚠️ 学習率が大きいと発散することがあります
              </div>
            )}
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
          <Button onClick={reset} variant="danger" className="w-full">↺ リセット（初期値を再抽選）</Button>

          {/* Param comparison */}
          <Card title="パラメータ推定値">
            <div className="space-y-1.5">
              <div className="grid grid-cols-3 gap-2 text-[10px] uppercase tracking-wider text-slate-400 pb-1 border-b border-slate-200">
                <span>θ</span><span className="text-right">真値</span><span className="text-right">推定</span>
              </div>
              {labels.map((lab, j) => (
                <div key={j} className="grid grid-cols-3 gap-2 text-xs font-mono">
                  <span className="text-slate-500">{lab}</span>
                  <span className="text-right text-slate-600">{config.theta[j].toFixed(2)}</span>
                  <span className="text-right text-indigo-600 font-semibold">{snap ? snap.theta[j].toFixed(2) : '—'}</span>
                </div>
              ))}
            </div>
          </Card>
        </div>

        {/* Plots */}
        <div className="space-y-5">
          <Card title="フィッティングの様子" right={
            <span className="text-xs font-mono text-indigo-600">Step {cur} / {hist.length - 1}</span>
          }>
            {diverged && (
              <div className="mb-3 rounded-lg border border-red-200 bg-red-50 p-2.5 text-xs text-red-700">
                🔥 発散しました。学習率を下げてリセットしてください。
              </div>
            )}
            <div className="h-80">
              <Plot
                data={[
                  { x: data.map(p => p.x), y: data.map(p => p.y), mode: 'markers',
                    marker: { color: '#475569', size: 5, opacity: 0.65 }, type: 'scatter', name: 'データ' },
                  { x: trueCurve.x, y: trueCurve.y, mode: 'lines',
                    line: { color: '#94a3b8', width: 2, dash: 'dash' }, type: 'scatter', name: '真の関数' },
                  { x: fitCurve.x, y: fitCurve.y, mode: 'lines',
                    line: { color: '#4f46e5', width: 3 }, type: 'scatter', name: '推定' },
                ] as Plotly.Data[]}
                layout={{ ...darkLayout('x', 'y'), height: 310 }}
                config={plotConfig}
              />
            </div>
            <p className="text-xs text-slate-500 mt-2">
              <span className="text-slate-400">— —</span> 真の関数（破線）
              <span className="text-indigo-600">———</span> 勾配降下法による推定（実線）
            </p>
          </Card>

          <Card title="損失（MSE）の推移">
            <div className="h-44">
              <Plot
                data={[{
                  x: hist.slice(0, cur + 1).map((_, i) => i),
                  y: hist.slice(0, cur + 1).map(h => (isFinite(h.loss) ? h.loss : null)),
                  mode: 'lines', line: { color: diverged ? '#dc2626' : '#059669', width: 2 },
                  fill: 'tozeroy', fillcolor: diverged ? 'rgba(220,38,38,0.08)' : 'rgba(5,150,105,0.08)',
                  connectgaps: false, type: 'scatter',
                }] as Plotly.Data[]}
                layout={{ ...darkLayout('イテレーション', 'MSE'), height: 175 }}
                config={plotConfig}
              />
            </div>
          </Card>

          {(status === 'paused' || status === 'done') && hist.length > 1 && (
            <Card>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-medium text-slate-600">ステップをスクラブ</span>
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

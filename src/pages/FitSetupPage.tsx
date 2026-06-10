import { useMemo } from 'react';
import Plot from '../components/Plot';
import { ParamControl, Button, Card, StepHeader } from '../components/ui';
import { darkLayout, plotConfig } from '../components/plotTheme';
import { FIT_LABEL, makeModel, sampleModel, defaultTheta } from '../utils/functions';
import type { FitFunctionType } from '../utils/functions';

export interface FitConfig {
  type: FitFunctionType;
  k: number;        // components for gmm
  theta: number[];  // TRUE parameters
  n: number;
  noise: number;
}

interface Props {
  config: FitConfig;
  setConfig: (c: FitConfig) => void;
  onBack: () => void;
  onNext: () => void;
}

const TYPES: FitFunctionType[] = ['line', 'trig', 'gaussian', 'gmm'];

export default function FitSetupPage({ config, setConfig, onBack, onNext }: Props) {
  const model = useMemo(() => makeModel(config.type, config.k), [config.type, config.k]);
  const fields = model.fields();

  const setType = (type: FitFunctionType) => {
    const k = config.k;
    setConfig({ ...config, type, theta: defaultTheta(type, k) });
  };
  const setK = (k: number) => {
    setConfig({ ...config, k, theta: defaultTheta('gmm', k) });
  };
  const setParam = (idx: number, v: number) => {
    const theta = [...config.theta];
    theta[idx] = v;
    setConfig({ ...config, theta });
  };

  // Preview of the true (noise-free) curve
  const curve = useMemo(() => sampleModel(model, config.theta, 240), [model, config.theta]);

  return (
    <div className="min-h-screen px-6 py-6 max-w-6xl mx-auto">
      <StepHeader
        title="関数フィット — データ設定"
        steps={['タスク選択', 'データ設定', '勾配降下法']}
        current={1}
        onBack={onBack}
      />

      <div className="grid lg:grid-cols-2 gap-6 items-start">
        {/* Left: configuration */}
        <div className="space-y-5">
          <Card title="① 関数形を選ぶ">
            <div className="grid grid-cols-2 gap-2">
              {TYPES.map(t => (
                <button
                  key={t}
                  onClick={() => setType(t)}
                  className={`py-2.5 px-2 rounded-lg text-xs font-medium text-left transition-all border ${
                    config.type === t
                      ? 'bg-indigo-600 border-indigo-600 text-white'
                      : 'bg-white border-slate-300 text-slate-600 hover:border-slate-400'
                  }`}
                >
                  {FIT_LABEL[t]}
                </button>
              ))}
            </div>

            {config.type === 'gmm' && (
              <div className="mt-4">
                <label className="block text-xs text-slate-600 mb-1.5">潜在変数の次元 k（成分数）</label>
                <div className="flex gap-2">
                  {[2, 3, 4].map(kk => (
                    <button
                      key={kk}
                      onClick={() => setK(kk)}
                      className={`flex-1 py-1.5 rounded-lg text-sm font-bold border transition-all ${
                        config.k === kk
                          ? 'bg-indigo-600 border-indigo-600 text-white'
                          : 'bg-white border-slate-300 text-slate-600 hover:border-slate-400'
                      }`}
                    >{kk}</button>
                  ))}
                </div>
              </div>
            )}
          </Card>

          {/* ② データ生成設定（旧③） */}
          <Card title="② データ生成設定">
            <ParamControl label="データ件数 N" value={config.n} min={50} max={500} step={10}
              onChange={v => setConfig({ ...config, n: Math.round(v) })} accent="emerald" />
            <ParamControl label="ノイズ強度 σ" value={config.noise} min={0.01} max={1} step={0.01}
              onChange={v => setConfig({ ...config, noise: v })} accent="emerald" />
          </Card>

          {/* ③ 真のパラメータ（旧②）— プレビューを見ながら調整 */}
          <Card title="③ 真のパラメータ θ を設定">
            <div className="max-h-80 overflow-y-auto pr-1">
              {fields.map((f, i) => (
                <ParamControl
                  key={i}
                  label={f.label}
                  value={config.theta[i]}
                  min={f.min} max={f.max} step={f.step}
                  onChange={v => setParam(i, v)}
                />
              ))}
            </div>
            <p className="text-xs text-slate-500 mt-1 leading-relaxed">
              ここで決めた θ が「真の関数」です。右のプレビューを見ながら調整できます。勾配降下法はこの θ を
              ランダムな初期値から推定します。
            </p>
          </Card>
        </div>

        {/* Right: preview + next (sticky so it stays visible while editing params) */}
        <div className="space-y-5 lg:sticky lg:top-6">
          <Card title="真の関数プレビュー">
            <div className="h-80">
              <Plot
                data={[{
                  x: curve.x, y: curve.y, mode: 'lines',
                  line: { color: '#4f46e5', width: 3 }, type: 'scatter',
                }] as Plotly.Data[]}
                layout={{ ...darkLayout('x', 'y'), height: 310 }}
                config={plotConfig}
              />
            </div>
            <p className="text-xs text-slate-500 mt-2 font-mono break-all">
              θ = [{config.theta.map(v => v.toFixed(2)).join(', ')}]
            </p>
          </Card>

          <div className="flex justify-end">
            <Button onClick={onNext}>このデータで勾配降下法へ →</Button>
          </div>
        </div>
      </div>
    </div>
  );
}

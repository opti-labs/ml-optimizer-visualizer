import { useMemo } from 'react';
import Plot from '../components/Plot';
import { ParamControl, Button, Card, StepHeader } from '../components/ui';
import { darkLayout, plotConfig, COLORS } from '../components/plotTheme';
import type { DataPoint } from '../utils/dataGen';
import { generateClusterData } from '../utils/dataGen';

export interface ClusterConfig {
  k: number;
  n: number;
  noise: number;
}

interface Props {
  config: ClusterConfig;
  setConfig: (c: ClusterConfig) => void;
  onBack: () => void;
  onNext: (points: DataPoint[]) => void;
}

export default function ClusterSetupPage({ config, setConfig, onBack, onNext }: Props) {
  // Generate a dataset for this config; reused as the run dataset on "next".
  const dataset = useMemo(
    () => generateClusterData(config.k, config.n, config.noise).points,
    [config.k, config.n, config.noise]
  );

  return (
    <div className="min-h-screen px-6 py-6 max-w-6xl mx-auto">
      <StepHeader
        title="クラスタリング — データ設定"
        steps={['タスク選択', 'データ設定', 'K平均法']}
        current={1}
        onBack={onBack}
      />

      <div className="grid lg:grid-cols-2 gap-6">
        <div className="space-y-5">
          <Card title="クラスター数 K">
            <div className="flex gap-2">
              {[2, 3, 4].map(kk => (
                <button
                  key={kk}
                  onClick={() => setConfig({ ...config, k: kk })}
                  className={`flex-1 py-2 rounded-lg text-sm font-bold border transition-all ${
                    config.k === kk
                      ? 'bg-emerald-600 border-emerald-500 text-white'
                      : 'bg-slate-700/50 border-slate-600 text-slate-400 hover:border-slate-500'
                  }`}
                >{kk}</button>
              ))}
            </div>
            <p className="text-xs text-slate-500 mt-3 leading-relaxed">
              2次元平面に K 個のガウス分布の塊からなるデータを生成します。K平均法はこの K を使って
              ランダムな初期重心を置き、各重心をクラスター中心へ動かしていきます。
            </p>
          </Card>

          <Card title="データ生成設定">
            <ParamControl label="データ件数 N" value={config.n} min={50} max={500} step={10}
              onChange={v => setConfig({ ...config, n: Math.round(v) })} accent="emerald" />
            <ParamControl label="ノイズ強度 σ" value={config.noise} min={0.05} max={1.2} step={0.05}
              onChange={v => setConfig({ ...config, noise: v })} accent="emerald" />
          </Card>

          <div className="flex justify-end">
            <Button onClick={() => onNext(dataset)}>このデータでK平均法へ →</Button>
          </div>
        </div>

        <Card title="生成データのプレビュー（真のクラスター）">
          <div className="h-96">
            <Plot
              data={[{
                x: dataset.map(p => p.x),
                y: dataset.map(p => p.y),
                mode: 'markers',
                marker: { color: dataset.map(p => COLORS[(p.label ?? 0) % COLORS.length]), size: 6, opacity: 0.7 },
                type: 'scatter',
              }] as Plotly.Data[]}
              layout={{ ...darkLayout('x₁', 'x₂'), height: 370 }}
              config={plotConfig}
            />
          </div>
        </Card>
      </div>
    </div>
  );
}

import { useState } from 'react';
import HomePage from './pages/HomePage';
import FitSetupPage from './pages/FitSetupPage';
import type { FitConfig } from './pages/FitSetupPage';
import FitRunPage from './pages/FitRunPage';
import ClusterSetupPage from './pages/ClusterSetupPage';
import type { ClusterConfig } from './pages/ClusterSetupPage';
import ClusterRunPage from './pages/ClusterRunPage';
import type { DataPoint } from './utils/dataGen';
import { defaultTheta } from './utils/functions';
import { makeModel } from './utils/functions';
import { generateFitData } from './utils/fitGD';

type Screen = 'home' | 'fit-setup' | 'fit-run' | 'cluster-setup' | 'cluster-run';

export default function App() {
  const [screen, setScreen] = useState<Screen>('home');

  // ─ Fit task ──────────────────────────────────────────────────────────────
  const [fitCfg, setFitCfg] = useState<FitConfig>({
    type: 'line', k: 3, theta: defaultTheta('line', 3), n: 150, noise: 0.2,
  });
  const [fitData, setFitData] = useState<DataPoint[]>([]);

  // ─ Cluster task ──────────────────────────────────────────────────────────
  const [clusterCfg, setClusterCfg] = useState<ClusterConfig>({ k: 3, n: 150, noise: 0.3 });
  const [clusterData, setClusterData] = useState<DataPoint[]>([]);

  const startFit = () => {
    const model = makeModel(fitCfg.type, fitCfg.k);
    setFitData(generateFitData(model, fitCfg.theta, fitCfg.n, fitCfg.noise));
    setScreen('fit-run');
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-700">
      {screen === 'home' && (
        <HomePage onSelect={t => setScreen(t === 'fit' ? 'fit-setup' : 'cluster-setup')} />
      )}

      {screen === 'fit-setup' && (
        <FitSetupPage
          config={fitCfg}
          setConfig={setFitCfg}
          onBack={() => setScreen('home')}
          onNext={startFit}
        />
      )}

      {screen === 'fit-run' && (
        <FitRunPage config={fitCfg} data={fitData} onBack={() => setScreen('fit-setup')} />
      )}

      {screen === 'cluster-setup' && (
        <ClusterSetupPage
          config={clusterCfg}
          setConfig={setClusterCfg}
          onBack={() => setScreen('home')}
          onNext={pts => { setClusterData(pts); setScreen('cluster-run'); }}
        />
      )}

      {screen === 'cluster-run' && (
        <ClusterRunPage k={clusterCfg.k} data={clusterData} onBack={() => setScreen('cluster-setup')} />
      )}
    </div>
  );
}

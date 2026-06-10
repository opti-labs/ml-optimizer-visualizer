import { useState } from 'react';
import HomePage from './pages/HomePage';
import type { TaskKey } from './pages/HomePage';
import FitSetupPage from './pages/FitSetupPage';
import type { FitConfig } from './pages/FitSetupPage';
import FitRunPage from './pages/FitRunPage';
import ClusterSetupPage from './pages/ClusterSetupPage';
import type { ClusterConfig } from './pages/ClusterSetupPage';
import ClusterRunPage from './pages/ClusterRunPage';
import LossLandscapeMode from './components/LossLandscapeMode';
import NeuronDecompositionMode from './components/NeuronDecompositionMode';
import OverfittingMode from './components/OverfittingMode';
import KernelTrickMode from './components/KernelTrickMode';
import type { DataPoint } from './utils/dataGen';
import { defaultTheta } from './utils/functions';
import { makeModel } from './utils/functions';
import { generateFitData } from './utils/fitGD';

type Screen =
  | 'home'
  | 'fit-setup' | 'fit-run'
  | 'cluster-setup' | 'cluster-run'
  | 'landscape' | 'neurons' | 'overfit' | 'kernel';

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

  const handleSelect = (task: TaskKey) => {
    switch (task) {
      case 'fit': setScreen('fit-setup'); break;
      case 'cluster': setScreen('cluster-setup'); break;
      case 'landscape': setScreen('landscape'); break;
      case 'neurons': setScreen('neurons'); break;
      case 'overfit': setScreen('overfit'); break;
      case 'kernel': setScreen('kernel'); break;
    }
  };

  const goHome = () => setScreen('home');

  return (
    <div className="min-h-screen bg-slate-50 text-slate-700">
      {screen === 'home' && <HomePage onSelect={handleSelect} />}

      {screen === 'fit-setup' && (
        <FitSetupPage
          config={fitCfg}
          setConfig={setFitCfg}
          onBack={goHome}
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
          onBack={goHome}
          onNext={pts => { setClusterData(pts); setScreen('cluster-run'); }}
        />
      )}
      {screen === 'cluster-run' && (
        <ClusterRunPage k={clusterCfg.k} data={clusterData} onBack={() => setScreen('cluster-setup')} />
      )}

      {screen === 'landscape' && <LossLandscapeMode onBack={goHome} />}
      {screen === 'neurons' && <NeuronDecompositionMode onBack={goHome} />}
      {screen === 'overfit' && <OverfittingMode onBack={goHome} />}
      {screen === 'kernel' && <KernelTrickMode onBack={goHome} />}
    </div>
  );
}

import { TrendingUp, Circle } from 'lucide-react';

export default function HomePage({ onSelect }: { onSelect: (task: 'fit' | 'cluster') => void }) {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-6 py-12">
      <div className="text-center mb-12">
        <h1 className="text-3xl font-bold text-slate-900 tracking-tight mb-3">
          機械学習 最適化ビジュアライザー
        </h1>
        <p className="text-slate-500 max-w-xl mx-auto leading-relaxed">
          学習したいタスクを選んでください。データを生成し、最適化アルゴリズムが
          解を求めていく過程をステップごとに可視化します。
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-6 w-full max-w-3xl">
        <button
          onClick={() => onSelect('fit')}
          className="group rounded-2xl border border-slate-200 bg-white shadow-sm hover:shadow-md hover:border-indigo-300 transition-all p-8 text-left"
        >
          <div className="w-12 h-12 rounded-xl bg-indigo-50 border border-indigo-200 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
            <TrendingUp className="text-indigo-600" size={24} />
          </div>
          <h2 className="text-lg font-bold text-slate-900 mb-2">関数フィット（回帰）</h2>
          <p className="text-sm text-slate-500 leading-relaxed mb-4">
            直線・三角関数・ガウス分布・混合ガウス分布の真のパラメータを設定し、
            ノイズ入りデータに対して<strong className="text-slate-700">勾配降下法</strong>でパラメータを推定します。
          </p>
          <span className="text-xs text-indigo-600 font-medium">勾配法でフィット →</span>
        </button>

        <button
          onClick={() => onSelect('cluster')}
          className="group rounded-2xl border border-slate-200 bg-white shadow-sm hover:shadow-md hover:border-emerald-300 transition-all p-8 text-left"
        >
          <div className="w-12 h-12 rounded-xl bg-emerald-50 border border-emerald-200 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
            <Circle className="text-emerald-600" size={24} />
          </div>
          <h2 className="text-lg font-bold text-slate-900 mb-2">クラスタリング</h2>
          <p className="text-sm text-slate-500 leading-relaxed mb-4">
            2次元のガウス混合データを生成し、<strong className="text-slate-700">K平均法</strong>で
            ランダムな初期重心がクラスター中心へ移動していく様子を可視化します。
          </p>
          <span className="text-xs text-emerald-600 font-medium">K平均法でクラスタリング →</span>
        </button>
      </div>
    </div>
  );
}

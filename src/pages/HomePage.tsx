import { TrendingUp, Circle, Mountain, Network, Activity, Orbit } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';

export type TaskKey = 'fit' | 'cluster' | 'landscape' | 'neurons' | 'overfit' | 'kernel';

interface CardDef {
  key: TaskKey;
  icon: LucideIcon;
  title: string;
  desc: React.ReactNode;
  tag: string;
  // Tailwind classes must be literal strings (no interpolation)
  iconWrap: string;
  iconColor: string;
  hoverBorder: string;
  tagColor: string;
}

const CARDS: CardDef[] = [
  {
    key: 'fit', icon: TrendingUp, title: '関数フィット（回帰）',
    desc: <>直線・三角関数・ガウス分布などの真のパラメータを設定し、<strong className="text-slate-700">勾配降下法</strong>で推定します。</>,
    tag: '勾配法でフィット →',
    iconWrap: 'bg-indigo-50 border-indigo-200', iconColor: 'text-indigo-600',
    hoverBorder: 'hover:border-indigo-300', tagColor: 'text-indigo-600',
  },
  {
    key: 'cluster', icon: Circle, title: 'クラスタリング',
    desc: <>2次元のガウス混合データを<strong className="text-slate-700">K平均法</strong>で分類。重心が動いていく様子を観察します。</>,
    tag: 'K平均法でクラスタリング →',
    iconWrap: 'bg-emerald-50 border-emerald-200', iconColor: 'text-emerald-600',
    hoverBorder: 'hover:border-emerald-300', tagColor: 'text-emerald-600',
  },
  {
    key: 'landscape', icon: Mountain, title: '3D損失ランドスケープ',
    desc: <>損失関数の「3Dの谷」を描画し、<strong className="text-slate-700">SGDとAdam</strong>がボールのように転がり落ちる軌跡を対比します。</>,
    tag: '谷を転がるパラメータ →',
    iconWrap: 'bg-violet-50 border-violet-200', iconColor: 'text-violet-600',
    hoverBorder: 'hover:border-violet-300', tagColor: 'text-violet-600',
  },
  {
    key: 'neurons', icon: Network, title: 'NN：隠れ層の役割分担',
    desc: <>複雑な関数を<strong className="text-slate-700">各ニューロンがどう分担して</strong>表現するのかを、個別のミニグラフで分解して可視化します。</>,
    tag: 'ニューロンの分業を見る →',
    iconWrap: 'bg-sky-50 border-sky-200', iconColor: 'text-sky-600',
    hoverBorder: 'hover:border-sky-300', tagColor: 'text-sky-600',
  },
  {
    key: 'overfit', icon: Activity, title: '過学習と正則化',
    desc: <>少ないデータ＋高次多項式で<strong className="text-slate-700">過学習</strong>を起こし、L2正則化λで滑らかな曲線に戻る様子を体験します。</>,
    tag: 'U字カーブを体験 →',
    iconWrap: 'bg-amber-50 border-amber-200', iconColor: 'text-amber-600',
    hoverBorder: 'hover:border-amber-300', tagColor: 'text-amber-600',
  },
  {
    key: 'kernel', icon: Orbit, title: 'SVMとカーネル・トリック',
    desc: <>直線では分離できないドーナツ型データを<strong className="text-slate-700">3次元へ写像</strong>し、平面で分離できるようになる様子を可視化します。</>,
    tag: '高次元への写像を見る →',
    iconWrap: 'bg-rose-50 border-rose-200', iconColor: 'text-rose-600',
    hoverBorder: 'hover:border-rose-300', tagColor: 'text-rose-600',
  },
];

export default function HomePage({ onSelect }: { onSelect: (task: TaskKey) => void }) {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-6 py-12">
      <div className="text-center mb-10">
        <h1 className="text-3xl font-bold text-slate-900 tracking-tight mb-3">
          機械学習 最適化ビジュアライザー
        </h1>
        <p className="text-slate-500 max-w-xl mx-auto leading-relaxed">
          学習したいタスクを選んでください。データを生成し、アルゴリズムが
          解を求めていく過程をステップごとに可視化します。
        </p>
      </div>

      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-5 w-full max-w-5xl">
        {CARDS.map(c => (
          <button
            key={c.key}
            onClick={() => onSelect(c.key)}
            className={`group rounded-2xl border border-slate-200 bg-white shadow-sm hover:shadow-md ${c.hoverBorder} transition-all p-6 text-left`}
          >
            <div className={`w-11 h-11 rounded-xl border flex items-center justify-center mb-3 group-hover:scale-110 transition-transform ${c.iconWrap}`}>
              <c.icon className={c.iconColor} size={22} />
            </div>
            <h2 className="text-base font-bold text-slate-900 mb-1.5">{c.title}</h2>
            <p className="text-xs text-slate-500 leading-relaxed mb-3">{c.desc}</p>
            <span className={`text-xs font-medium ${c.tagColor}`}>{c.tag}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

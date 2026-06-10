import type { ReactNode } from 'react';

// ── Param control: slider + exact numeric input ──────────────────────────────
export function ParamControl({
  label, value, min, max, step, onChange, accent = 'indigo',
}: {
  label: string; value: number; min: number; max: number; step: number;
  onChange: (v: number) => void; accent?: string;
}) {
  const accentMap: Record<string, string> = {
    indigo: 'accent-indigo-600', emerald: 'accent-emerald-600', orange: 'accent-orange-500',
  };
  return (
    <div className="mb-3">
      <div className="flex items-center justify-between mb-1 gap-2">
        <label className="text-xs text-slate-600">{label}</label>
        <input
          type="number" value={value} min={min} max={max} step={step}
          onChange={e => onChange(clampNum(Number(e.target.value), min, max))}
          className="w-20 rounded-md border border-slate-300 bg-white px-2 py-0.5 text-xs text-right font-mono text-indigo-700 focus:border-indigo-500 focus:outline-none"
        />
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(Number(e.target.value))}
        className={`w-full h-1.5 appearance-none rounded-full bg-slate-200 cursor-pointer ${accentMap[accent] || accentMap.indigo}`}
      />
    </div>
  );
}

function clampNum(v: number, lo: number, hi: number) {
  if (Number.isNaN(v)) return lo;
  return Math.max(lo, Math.min(hi, v));
}

// ── Buttons ──────────────────────────────────────────────────────────────────
export function Button({
  children, onClick, variant = 'primary', disabled, className = '',
}: {
  children: ReactNode; onClick?: () => void;
  variant?: 'primary' | 'ghost' | 'amber' | 'danger'; disabled?: boolean; className?: string;
}) {
  const base = 'rounded-xl text-sm font-semibold transition-all flex items-center justify-center gap-2 px-4 py-2.5 border';
  const styles: Record<string, string> = {
    primary: 'bg-indigo-600 hover:bg-indigo-500 text-white border-indigo-600 shadow-sm shadow-indigo-200',
    ghost: 'bg-white hover:bg-slate-100 text-slate-700 border-slate-300',
    amber: 'bg-amber-500 hover:bg-amber-400 text-white border-amber-500',
    danger: 'bg-white hover:bg-slate-100 text-slate-500 border-slate-300',
  };
  const dis = 'opacity-40 cursor-not-allowed pointer-events-none';
  return (
    <button onClick={onClick} disabled={disabled}
      className={`${base} ${styles[variant]} ${disabled ? dis : ''} ${className}`}>
      {children}
    </button>
  );
}

// ── Card ─────────────────────────────────────────────────────────────────────
export function Card({ title, children, right }: { title?: ReactNode; children: ReactNode; right?: ReactNode }) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white shadow-sm overflow-hidden">
      {title && (
        <div className="px-4 py-2.5 border-b border-slate-100 flex items-center justify-between bg-slate-50/60">
          <span className="text-sm font-semibold text-slate-700">{title}</span>
          {right}
        </div>
      )}
      <div className="p-4">{children}</div>
    </div>
  );
}

// ── Step indicator ───────────────────────────────────────────────────────────
export function StepHeader({
  steps, current, onBack, title,
}: { steps: string[]; current: number; onBack?: () => void; title: string }) {
  return (
    <div className="flex items-center gap-4 mb-6">
      {onBack && (
        <button onClick={onBack}
          className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-xs text-slate-600 hover:border-slate-400 hover:bg-slate-100 transition-colors">
          ← 戻る
        </button>
      )}
      <h1 className="text-lg font-bold text-slate-900 tracking-tight">{title}</h1>
      <div className="ml-auto flex items-center gap-1.5">
        {steps.map((s, i) => (
          <div key={i} className="flex items-center gap-1.5">
            <span className={`flex items-center gap-1.5 text-xs ${i === current ? 'text-indigo-600' : 'text-slate-400'}`}>
              <span className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold border ${
                i === current ? 'bg-indigo-600 border-indigo-600 text-white'
                  : i < current ? 'bg-slate-200 border-slate-300 text-slate-500' : 'border-slate-300 text-slate-400'
              }`}>{i + 1}</span>
              <span className="hidden md:inline">{s}</span>
            </span>
            {i < steps.length - 1 && <span className="text-slate-300">/</span>}
          </div>
        ))}
      </div>
    </div>
  );
}

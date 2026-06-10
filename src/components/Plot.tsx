// react-plotly.js' default export breaks under Vite/rolldown ESM interop
// (resolves to an object, not a component). The factory approach is the
// documented workaround: build the component ourselves from a Plotly bundle.
//
// Note: under Vite's CJS→ESM interop the factory's default export gets
// double-wrapped as { default: fn }, so we unwrap defensively to get the
// actual factory function regardless of how the interop resolves it.
import { useEffect, useRef } from 'react';
import Plotly from 'plotly.js-dist-min';
import factoryImport from 'react-plotly.js/factory';

type Factory = (p: typeof Plotly) => React.ComponentType<Record<string, unknown>>;

const createPlotlyComponent =
  (typeof factoryImport === 'function'
    ? factoryImport
    : (factoryImport as { default: Factory }).default) as Factory;

const RawPlot = createPlotlyComponent(Plotly);

/**
 * Resize-robust wrapper. react-plotly.js measures the container width once at
 * mount; if the flex/grid layout hasn't settled yet (or the panel was hidden),
 * it captures width 0 and the chart renders invisibly until a window resize.
 * We force a Plotly resize whenever the container's own box changes via a
 * ResizeObserver (and enable useResizeHandler for window resizes), so the plot
 * always fills its container regardless of mount timing.
 */
export default function Plot(props: Record<string, unknown>) {
  const wrapRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const wrap = wrapRef.current;
    if (!wrap || typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(() => {
      const gd = wrap.querySelector('.js-plotly-plot') as HTMLElement | null;
      if (gd && (gd as unknown as { _fullLayout?: unknown })._fullLayout) {
        try { Plotly.Plots.resize(gd); } catch { /* ignore */ }
      }
    });
    ro.observe(wrap);
    return () => ro.disconnect();
  }, []);

  const style = { width: '100%', ...((props.style as object) ?? {}) };
  return (
    <div ref={wrapRef} style={{ width: '100%' }}>
      <RawPlot {...props} useResizeHandler style={style} />
    </div>
  );
}

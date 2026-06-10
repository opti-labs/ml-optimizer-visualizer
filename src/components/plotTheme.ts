// Shared Plotly light-theme helpers.
export const COLORS = ['#4f46e5', '#059669', '#ea580c', '#db2777', '#2563eb'];

export const darkLayout = (
  xTitle?: string,
  yTitle?: string,
  extra: Partial<Plotly.Layout> = {}
): Partial<Plotly.Layout> => ({
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(248,250,252,0.8)',
  font: { color: '#475569', size: 11 },
  // fixedrange: true disables drag/scroll zoom & pan on both axes.
  xaxis: {
    title: xTitle ? { text: xTitle, font: { size: 11 } } : undefined,
    gridcolor: '#e2e8f0', zerolinecolor: '#cbd5e1', color: '#64748b', tickfont: { size: 10 },
    fixedrange: true,
  },
  yaxis: {
    title: yTitle ? { text: yTitle, font: { size: 11 } } : undefined,
    gridcolor: '#e2e8f0', zerolinecolor: '#cbd5e1', color: '#64748b', tickfont: { size: 10 },
    fixedrange: true,
  },
  margin: { l: 48, r: 16, t: 16, b: 42 },
  showlegend: false,
  dragmode: false,
  ...extra,
});

export const plotConfig: Partial<Plotly.Config> = {
  displayModeBar: false,
  responsive: true,
  scrollZoom: false,
  doubleClick: false,
};

// ── 3D scene layout (loss landscape / kernel trick) ──────────────────────────
// 3D plots keep orbit-rotation enabled (it's essential for understanding the
// geometry) but inherit the same light styling as the 2D charts.
const ax3 = (title: string) => ({
  title: { text: title, font: { size: 11 } },
  gridcolor: '#e2e8f0',
  zerolinecolor: '#cbd5e1',
  color: '#64748b',
  showbackground: true,
  backgroundcolor: 'rgba(248,250,252,0.85)',
  tickfont: { size: 9 },
});

export const layout3d = (
  xTitle: string, yTitle: string, zTitle: string,
  extra: Partial<Plotly.Layout> = {}
): Partial<Plotly.Layout> => ({
  paper_bgcolor: 'rgba(0,0,0,0)',
  font: { color: '#475569', size: 11 },
  margin: { l: 0, r: 0, t: 0, b: 0 },
  showlegend: false,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  scene: {
    xaxis: ax3(xTitle), yaxis: ax3(yTitle), zaxis: ax3(zTitle),
    camera: { eye: { x: 1.55, y: -1.6, z: 0.9 } },
  } as any,
  ...extra,
});

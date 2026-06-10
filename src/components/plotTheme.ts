// Shared Plotly dark-theme helpers.
export const COLORS = ['#818cf8', '#34d399', '#fb923c', '#f472b6', '#60a5fa'];

export const darkLayout = (
  xTitle?: string,
  yTitle?: string,
  extra: Partial<Plotly.Layout> = {}
): Partial<Plotly.Layout> => ({
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(15,17,30,0.6)',
  font: { color: '#94a3b8', size: 11 },
  xaxis: {
    title: xTitle ? { text: xTitle, font: { size: 11 } } : undefined,
    gridcolor: '#1e2235', zerolinecolor: '#2d3252', color: '#64748b', tickfont: { size: 10 },
  },
  yaxis: {
    title: yTitle ? { text: yTitle, font: { size: 11 } } : undefined,
    gridcolor: '#1e2235', zerolinecolor: '#2d3252', color: '#64748b', tickfont: { size: 10 },
  },
  margin: { l: 48, r: 16, t: 16, b: 42 },
  showlegend: false,
  ...extra,
});

export const plotConfig: Partial<Plotly.Config> = {
  displayModeBar: false,
  responsive: true,
};

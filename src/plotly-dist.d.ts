// plotly.js-dist-min ships no types of its own, but its runtime API matches
// the full plotly.js bundle covered by @types/plotly.js.
declare module 'plotly.js-dist-min' {
  import * as Plotly from 'plotly.js';
  export = Plotly;
}

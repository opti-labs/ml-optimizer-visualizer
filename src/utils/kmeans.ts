import type { DataPoint } from './dataGen';

// ── K-means (Lloyd's algorithm) ──────────────────────────────────────────────

export interface KMeansParams {
  centroids: [number, number][];
}

export interface KMeansSnapshot {
  centroids: [number, number][];
  assignments: number[];   // nearest-centroid index for each point
  inertia: number;         // within-cluster sum of squared distances (objective)
  movement: number;        // total centroid shift on this step (→ 0 at convergence)
}

function sqDist(ax: number, ay: number, bx: number, by: number) {
  const dx = ax - bx, dy = ay - by;
  return dx * dx + dy * dy;
}

/** Random initial centroids: K distinct data points chosen at random (Forgy). */
export function initKMeans(k: number, points: DataPoint[]): KMeansParams {
  const n = points.length;
  const idx = new Set<number>();
  while (idx.size < Math.min(k, n)) idx.add(Math.floor(Math.random() * n));
  const centroids = [...idx].map(i => [points[i].x, points[i].y] as [number, number]);
  // pad (shouldn't happen for n >= k) just in case
  while (centroids.length < k) centroids.push([points[0].x, points[0].y]);
  return { centroids };
}

/** Assign every point to its nearest centroid. */
export function assignClusters(points: DataPoint[], centroids: [number, number][]): number[] {
  return points.map(p => {
    let best = 0, bestD = Infinity;
    for (let c = 0; c < centroids.length; c++) {
      const d = sqDist(p.x, p.y, centroids[c][0], centroids[c][1]);
      if (d < bestD) { bestD = d; best = c; }
    }
    return best;
  });
}

function computeInertia(points: DataPoint[], centroids: [number, number][], assignments: number[]): number {
  let s = 0;
  for (let i = 0; i < points.length; i++) {
    const c = centroids[assignments[i]];
    s += sqDist(points[i].x, points[i].y, c[0], c[1]);
  }
  return s;
}

/** Snapshot of the initial state (random centroids, first assignment). */
export function kmeansInitialSnapshot(points: DataPoint[], params: KMeansParams): KMeansSnapshot {
  const assignments = assignClusters(points, params.centroids);
  return {
    centroids: params.centroids.map(c => [...c] as [number, number]),
    assignments,
    inertia: computeInertia(points, params.centroids, assignments),
    movement: 0,
  };
}

/**
 * One Lloyd iteration:
 *   1. assign each point to the nearest current centroid,
 *   2. move each centroid to the mean of its assigned points.
 * The returned snapshot keeps the assignment used for the update (so points are
 * coloured by the clusters that produced the new centroid positions).
 */
export function kmeansStep(points: DataPoint[], params: KMeansParams): {
  params: KMeansParams;
  snapshot: KMeansSnapshot;
} {
  const k = params.centroids.length;
  const assignments = assignClusters(points, params.centroids);

  // objective with the centroids that were used for this assignment
  const inertia = computeInertia(points, params.centroids, assignments);

  const sumX = Array(k).fill(0), sumY = Array(k).fill(0), count = Array(k).fill(0);
  for (let i = 0; i < points.length; i++) {
    const c = assignments[i];
    sumX[c] += points[i].x;
    sumY[c] += points[i].y;
    count[c] += 1;
  }

  let movement = 0;
  const newCentroids = params.centroids.map((c, j) => {
    if (count[j] === 0) return [...c] as [number, number];  // keep empty cluster in place
    const nc: [number, number] = [sumX[j] / count[j], sumY[j] / count[j]];
    movement += Math.sqrt(sqDist(c[0], c[1], nc[0], nc[1]));
    return nc;
  });

  return {
    params: { centroids: newCentroids },
    snapshot: { centroids: newCentroids.map(c => [...c] as [number, number]), assignments, inertia, movement },
  };
}

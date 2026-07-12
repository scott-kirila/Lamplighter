import type { RunEpoch } from '../store/runStore'

// Fixed display order for metric keys (matches the generated train() report);
// any others follow, sorted.
const METRIC_ORDER = ['train_loss', 'train_acc', 'val_loss', 'val_acc']

// The metric columns present across a run's epochs, in display order — the known
// ones first (train before val), then any extras alphabetically. A metric that
// only some epochs carry (e.g. val without a val loader) still gets a column.
export function metricColumns(epochs: RunEpoch[]): string[] {
  const seen = new Set<string>()
  for (const e of epochs) for (const k of Object.keys(e.metrics)) seen.add(k)
  return [
    ...METRIC_ORDER.filter((k) => seen.has(k)),
    ...[...seen].filter((k) => !METRIC_ORDER.includes(k)).sort(),
  ]
}

// A metric value for the table (blank when the epoch doesn't carry it).
export const fmtMetric = (v: number | undefined): string => (v === undefined ? '' : v.toFixed(4))

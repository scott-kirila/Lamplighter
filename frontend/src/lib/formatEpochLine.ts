import type { RunEpoch } from '../store/graphStore'

// Fixed display order for metric keys (matches the generated train() report).
const METRIC_ORDER = ['train_loss', 'train_acc', 'val_loss', 'val_acc']

// One monospace progress line for a streamed epoch, e.g.
// "epoch  3/10  train_loss 0.4321  train_acc 0.812".
export function formatEpochLine(e: RunEpoch): string {
  const width = String(e.epochs).length
  const parts = [`epoch ${String(e.epoch).padStart(width)}/${e.epochs}`]
  const keys = [
    ...METRIC_ORDER.filter((k) => k in e.metrics),
    ...Object.keys(e.metrics).filter((k) => !METRIC_ORDER.includes(k)).sort(),
  ]
  for (const k of keys) parts.push(`${k} ${e.metrics[k].toFixed(4)}`)
  return parts.join('  ')
}

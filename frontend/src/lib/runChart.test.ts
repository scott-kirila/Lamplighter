import { describe, expect, it } from 'vitest'
import {
  chartDomain,
  comparisonCharts,
  discoverCharts,
  epochTicks,
  epochX,
  linearTicks,
  polylinePoints,
  seriesFor,
  tickLabel,
} from './runChart'
import type { RunEpoch } from '../store/graphStore'

const epoch = (n: number, metrics: Record<string, number>): RunEpoch => ({
  epoch: n,
  epochs: 10,
  metrics,
})

describe('seriesFor', () => {
  it('extracts values per key and drops absent metrics', () => {
    const epochs = [epoch(1, { train_loss: 1.0 }), epoch(2, { train_loss: 0.5 })]
    expect(seriesFor(epochs, ['train_loss', 'val_loss'])).toEqual([
      { key: 'train_loss', values: [1.0, 0.5] },
    ])
  })

  it('keeps series independent when a metric appears late', () => {
    const epochs = [epoch(1, { train_loss: 1 }), epoch(2, { train_loss: 0.5, val_loss: 0.7 })]
    expect(seriesFor(epochs, ['train_loss', 'val_loss'])).toEqual([
      { key: 'train_loss', values: [1, 0.5] },
      { key: 'val_loss', values: [0.7] },
    ])
  })
})

describe('discoverCharts', () => {
  it('groups supervised metrics into loss (train/val) and accuracy charts', () => {
    const epochs = [
      epoch(1, { train_loss: 1, val_loss: 0.9, train_acc: 0.4, val_acc: 0.45 }),
      epoch(2, { train_loss: 0.5, val_loss: 0.7, train_acc: 0.6, val_acc: 0.55 }),
    ]
    const charts = discoverCharts(epochs)
    expect(charts.map((c) => c.group)).toEqual(['loss', 'acc'])
    expect(charts[0].title).toBe('loss')
    expect(charts[1].title).toBe('accuracy')
    expect(charts[0].series.map((s) => s.label)).toEqual(['train', 'val'])
    expect(charts[0].series[1].values).toEqual([0.9, 0.7])
  })

  it('charts a GAN g_loss/d_loss together, no hardcoded keys', () => {
    const epochs = [epoch(1, { g_loss: 2, d_loss: 1 }), epoch(2, { g_loss: 1.5, d_loss: 0.8 })]
    const charts = discoverCharts(epochs)
    expect(charts).toHaveLength(1)
    expect(charts[0].group).toBe('loss')
    expect(charts[0].series.map((s) => s.label)).toEqual(['g', 'd'])
    expect(charts[0].series[0].values).toEqual([2, 1.5])
  })
})

describe('chartDomain', () => {
  it('pads min/max so lines never touch the edges', () => {
    const { min, max } = chartDomain([{ key: 'a', values: [0, 1] }])
    expect(min).toBeLessThan(0)
    expect(max).toBeGreaterThan(1)
  })

  it('widens a flat domain instead of degenerating', () => {
    const { min, max } = chartDomain([{ key: 'a', values: [0.5, 0.5] }])
    expect(max - min).toBeGreaterThan(0.5)
    expect(min).toBeLessThan(0.5)
    expect(max).toBeGreaterThan(0.5)
  })
})

describe('polylinePoints', () => {
  it('maps values across the planned-epoch width (curve grows rightward)', () => {
    // 2 of 5 planned epochs, domain [0, 1], canvas 100x100.
    const pts = polylinePoints([0, 1], 5, 0, 1, 100, 100)
    // x: epoch 1 at 0, epoch 2 at 1/4 of the width; y inverted (1 → top).
    expect(pts).toBe('0.00,100.00 25.00,0.00')
  })

  it('fills the full width when the run is complete', () => {
    const pts = polylinePoints([0, 0.5, 1], 3, 0, 1, 100, 100)
    expect(pts).toBe('0.00,100.00 50.00,50.00 100.00,0.00')
  })

  it('handles a single point without NaN', () => {
    expect(polylinePoints([0.5], 1, 0, 1, 100, 100)).toBe('0.00,50.00')
  })
})

describe('linearTicks', () => {
  it('spans min to max evenly', () => {
    expect(linearTicks(0, 3, 4)).toEqual([0, 1, 2, 3])
  })
})

describe('epochTicks', () => {
  it('uses step 1 for short runs', () => {
    expect(epochTicks(5)).toEqual([1, 2, 3, 4, 5])
  })

  it('picks nice steps to stay under the label budget', () => {
    expect(epochTicks(12)).toEqual([2, 4, 6, 8, 10, 12])
    expect(epochTicks(100)).toEqual([20, 40, 60, 80, 100])
    expect(epochTicks(30)).toEqual([5, 10, 15, 20, 25, 30])
  })

  it('handles a single epoch', () => {
    expect(epochTicks(1)).toEqual([1])
  })
})

describe('tickLabel', () => {
  it('keeps ~3 significant digits without trailing zeros', () => {
    expect(tickLabel(0.432149)).toBe('0.432')
    expect(tickLabel(1.5)).toBe('1.5')
    expect(tickLabel(0)).toBe('0')
    expect(tickLabel(123.456)).toBe('123')
  })
})

describe('epochX', () => {
  it('matches the polyline slot math (epoch 1 at 0, planned at full width)', () => {
    expect(epochX(1, 10, 90)).toBe(0)
    expect(epochX(10, 10, 90)).toBe(90)
    expect(epochX(2, 5, 100)).toBe(25)
  })
})

describe('comparisonCharts', () => {
  const live = [
    { epoch: 1, epochs: 3, metrics: { train_loss: 1.0, val_loss: 1.2 } },
    { epoch: 2, epochs: 3, metrics: { train_loss: 0.8, val_loss: 1.1 } },
  ]
  const stored = { name: 'run-a', history: { train_loss: [0.9, 0.7, 0.6], val_loss: [1.0, 0.9, 0.95] } }

  it('overlays a compared run onto the live series, grouped by metric', () => {
    const charts = comparisonCharts(live, [stored])
    expect(charts).toHaveLength(1) // everything is *_loss → one chart
    const labels = charts[0].series.map((s) => s.label)
    expect(labels).toEqual(['train', 'val', 'run-a·train', 'run-a·val'])
    // Unique keys (the name prefixes the metric), values straight from history.
    const compared = charts[0].series.find((s) => s.key === 'run-a:train_loss')!
    expect(compared.values).toEqual([0.9, 0.7, 0.6])
  })

  it('charts compared runs alone when nothing has streamed', () => {
    const charts = comparisonCharts([], [stored])
    expect(charts[0].series.map((s) => s.label)).toEqual(['run-a·train', 'run-a·val'])
  })

  it('groups a compared GAN with a live supervised run by suffix', () => {
    const gan = { name: 'gan', history: { g_loss: [1, 2], d_loss: [3, 4] } }
    const charts = comparisonCharts(live, [gan])
    expect(charts).toHaveLength(1) // all *_loss share the chart
    expect(charts[0].series.map((s) => s.label)).toEqual(['train', 'val', 'gan·g', 'gan·d'])
  })
})

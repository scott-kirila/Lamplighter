import { describe, expect, it } from 'vitest'
import {
  chartDomain,
  chartTicks,
  clampDomain,
  comparisonCharts,
  discoverCharts,
  epochTicks,
  epochX,
  linearTicks,
  logUsable,
  mergedLossSeries,
  metricGroup,
  polylinePoints,
  seriesFor,
  tickLabel,
  xyPolylinePoints,
} from './runChart'
import type { RunEpoch } from '../store/runStore'

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

describe('log scale', () => {
  it('computes the domain in log10 space over positive values only', () => {
    // The GAN case: one loss orders of magnitude below the other.
    const { min, max } = chartDomain([{ key: 'a', values: [1e-4, 10, 0, -1] }], 'log')
    // log10 range is [-4, 1] plus 8% padding — 0/-1 are excluded, not -Infinity.
    expect(min).toBeCloseTo(-4.4, 5)
    expect(max).toBeCloseTo(1.4, 5)
  })

  it('separates curves that flat-line together on a linear axis', () => {
    // Linear: 1e-4 and 1e-3 both map to the bottom ~0.1% of a [0,1] domain.
    // Log over [1e-4, 1]: they sit a fifth of the height apart.
    const log = polylinePoints([1e-4, 1e-3], 2, -4, 0, 100, 100, 'log')
    expect(log).toBe('0.00,100.00 100.00,75.00')
  })

  it('clamps non-positive values to the floor so the polyline stays connected', () => {
    const pts = polylinePoints([1, 0], 2, 0, 2, 100, 100, 'log')
    expect(pts).toBe('0.00,100.00 100.00,100.00')
  })

  it('log ticks label the real quantity (10^position), tiny ones exponential', () => {
    const ticks = chartTicks(-3, 0, 'log')
    expect(ticks.map((t) => t.label)).toEqual(['1.0e-3', '0.01', '0.1', '1'])
    expect(ticks[0].value).toBe(-3) // gridline positioned in domain space
  })

  it('linear ticks are unchanged by the wrapper', () => {
    expect(chartTicks(0, 3)).toEqual([
      { value: 0, label: '0' },
      { value: 1, label: '1' },
      { value: 2, label: '2' },
      { value: 3, label: '3' },
    ])
  })

  it('logUsable is false when nothing is positive (chart falls back to linear)', () => {
    expect(logUsable([{ key: 'a', values: [0, -1] }])).toBe(false)
    expect(logUsable([{ key: 'a', values: [0, 0.5] }])).toBe(true)
  })
})

describe('clampDomain', () => {
  it('caps a padded accuracy domain at the proportion bounds', () => {
    // acc reaching 1.0: the 8% pad would show a "1.01"-style top tick.
    const d = clampDomain(chartDomain([{ key: 'a', values: [0.4, 1.0] }]), 0, 1)
    expect(d.max).toBe(1)
    expect(d.min).toBeGreaterThan(0) // in-range padding is kept
    const low = clampDomain(chartDomain([{ key: 'a', values: [0.02, 0.4] }]), 0, 1)
    expect(low.min).toBe(0) // no negative accuracy either
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

  it('goes exponential below 0.01 so labels fit the chart gutter', () => {
    expect(tickLabel(0.000629)).toBe('6.3e-4') // was "0.000629" — clipped in the 48px gutter
    expect(tickLabel(-0.00391)).toBe('-3.9e-3')
    expect(tickLabel(0.0126)).toBe('0.0126') // at/above the threshold stays plain
  })
})

describe('metricGroup', () => {
  it('groups by the suffix after the first underscore (bare keys as themselves)', () => {
    expect(metricGroup('train_loss')).toBe('loss')
    expect(metricGroup('policy_loss')).toBe('loss')
    expect(metricGroup('mean_return')).toBe('return')
    expect(metricGroup('episode_return')).toBe('return') // the RL step stream routes here
    expect(metricGroup('entropy')).toBe('entropy')
  })
})

describe('epochX', () => {
  it('matches the polyline slot math (epoch 1 at 0, planned at full width)', () => {
    expect(epochX(1, 10, 90)).toBe(0)
    expect(epochX(10, 10, 90)).toBe(90)
    expect(epochX(2, 5, 100)).toBe(25)
  })
})

describe('merged loss chart (epoch-axis x)', () => {
  const ep = (epoch: number, metrics: Record<string, number>) => ({ epoch, metrics })

  it('layers each metric: faint raw step curve under its epoch-mean line', () => {
    const merged = mergedLossSeries(
      [ep(1, { train_loss: 1.0, val_loss: 1.1 }), ep(2, { train_loss: 0.5, val_loss: 0.7 })],
      ['train_loss', 'val_loss'],
      [
        { epoch_x: 0.5, metrics: { train_loss: 1.2 } },
        { epoch_x: 1.5, metrics: { train_loss: 0.6 } },
      ]
    )
    // Raw texture first (painted under), then the mean lines on top.
    expect(merged.map((s) => [s.key, s.raw ?? false])).toEqual([
      ['train_loss·steps', true],
      ['train_loss', false],
      ['val_loss', false],
    ])
    expect(merged[0].label).toBe('train') // raw + mean share the metric's label (one hue)
    expect(merged[0].points).toEqual([
      { x: 0.5, y: 1.2 },
      { x: 1.5, y: 0.6 },
    ])
    expect(merged[1].points).toEqual([
      { x: 1, y: 1.0 },
      { x: 2, y: 0.5 },
    ]) // the epoch mean survives — it carries the trend
    expect(merged[2].points).toEqual([
      { x: 1, y: 1.1 },
      { x: 2, y: 0.7 },
    ])
  })

  it('epoch overlays use TRUE epoch numbers — a resumed run does not restart at 1', () => {
    const merged = mergedLossSeries(
      [ep(5, { val_loss: 0.9 }), ep(6, { val_loss: 0.8 })], // resumed segment
      ['val_loss'],
      []
    )
    expect(merged[0].points).toEqual([
      { x: 5, y: 0.9 },
      { x: 6, y: 0.8 },
    ])
  })

  it('degrades to epoch-only series when steps carry no baked position', () => {
    const merged = mergedLossSeries(
      [ep(1, { train_loss: 1.0 }), ep(2, { train_loss: 0.5 })],
      ['train_loss'],
      [{ metrics: { train_loss: 1.2 } }] // IterableDataset: no epoch_x
    )
    expect(merged).toHaveLength(1)
    expect(merged[0].points).toEqual([
      { x: 1, y: 1.0 },
      { x: 2, y: 0.5 },
    ])
  })

  it('xyPolylinePoints maps epoch-axis x over [0, xMax] and y through the scale', () => {
    const pts = xyPolylinePoints([{ x: 1, y: 0 }, { x: 4, y: 1 }], 4, 0, 1, 100, 100)
    expect(pts).toBe('25.00,100.00 100.00,0.00')
    // Log path: non-positive clamps to the floor, positives transform.
    const log = xyPolylinePoints([{ x: 2, y: 1e-4 }, { x: 4, y: 1e-3 }], 4, -4, 0, 100, 100, 'log')
    expect(log).toBe('50.00,100.00 100.00,75.00')
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

import { describe, expect, it } from 'vitest'
import { fmtDuration, fmtMetric, metricColumns } from './epochMetrics'
import type { RunEpoch } from '../store/runStore'

const ep = (metrics: Record<string, number>): RunEpoch => ({ epoch: 1, epochs: 1, metrics })

describe('metricColumns', () => {
  it('orders known metrics canonically (train before val)', () => {
    expect(metricColumns([ep({ val_loss: 0.5, train_loss: 0.4 })])).toEqual(['train_loss', 'val_loss'])
  })

  it('appends unknown metrics alphabetically after the known ones', () => {
    expect(metricColumns([ep({ zeta: 1, train_loss: 2, alpha: 3 })])).toEqual(['train_loss', 'alpha', 'zeta'])
  })

  it('unions keys across epochs (val may appear only some epochs)', () => {
    expect(metricColumns([ep({ train_loss: 1 }), ep({ train_loss: 1, val_loss: 2 })])).toEqual([
      'train_loss',
      'val_loss',
    ])
  })

  it('is empty for metrics-less epochs', () => {
    expect(metricColumns([ep({})])).toEqual([])
  })
})

describe('fmtMetric', () => {
  it('formats to 4 dp, blank when absent', () => {
    expect(fmtMetric(0.5)).toBe('0.5000')
    expect(fmtMetric(undefined)).toBe('')
  })
})

describe('fmtDuration', () => {
  it('shows sub-minute as seconds with one decimal', () => {
    expect(fmtDuration(12.34)).toBe('12.3s')
    expect(fmtDuration(0)).toBe('0.0s')
  })

  it('shows a minute or more as m/ss, zero-padding the seconds', () => {
    expect(fmtDuration(60)).toBe('1m00s')
    expect(fmtDuration(123)).toBe('2m03s')
  })

  it('is blank when absent', () => {
    expect(fmtDuration(undefined)).toBe('')
  })
})

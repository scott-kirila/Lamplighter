import { describe, expect, it } from 'vitest'
import { formatEpochLine } from './formatEpochLine'

describe('formatEpochLine', () => {
  it('formats known metrics in canonical order with aligned epoch numbers', () => {
    expect(
      formatEpochLine({ epoch: 3, epochs: 10, metrics: { val_loss: 0.5, train_loss: 0.4321 } })
    ).toBe('epoch  3/10  train_loss 0.4321  val_loss 0.5000')
  })

  it('appends unknown metrics alphabetically after the known ones', () => {
    expect(
      formatEpochLine({ epoch: 1, epochs: 5, metrics: { zeta: 1, train_loss: 2 } })
    ).toBe('epoch 1/5  train_loss 2.0000  zeta 1.0000')
  })

  it('handles a metrics-less epoch', () => {
    expect(formatEpochLine({ epoch: 2, epochs: 2, metrics: {} })).toBe('epoch 2/2')
  })
})

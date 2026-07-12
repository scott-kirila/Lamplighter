// A tensor from /api/run/preview: a batched {shape:[n, …], data} pair (flat,
// row-major) plus a truncation flag for exotic huge per-sample sizes.
export interface TensorPayload {
  shape: number[]
  data: number[]
  truncated?: boolean
}

export type TensorKind = 'image' | 'bars' | 'scalar'

// The *per-sample* shape (batch dim already stripped) → how to draw it. Shape
// alone, no task knowledge: an HxW / 1xHxW / 3xHxW block is an image, a vector is
// bars, a lone value is a scalar. Anything else falls back to bars (flattened).
export function tensorKind(sampleShape: number[]): TensorKind {
  const numel = sampleShape.reduce((a, b) => a * b, 1)
  if (numel <= 1) return 'scalar'
  if (sampleShape.length === 2 && sampleShape[0] >= 2 && sampleShape[1] >= 2) return 'image'
  if (
    sampleShape.length === 3 &&
    (sampleShape[0] === 1 || sampleShape[0] === 3) &&
    sampleShape[1] >= 2 &&
    sampleShape[2] >= 2
  ) {
    return 'image'
  }
  return 'bars'
}

// Slice one example out of a batched tensor (strip the batch dim, take its window).
export function sampleAt(t: TensorPayload, index: number): { shape: number[]; data: number[] } {
  const shape = t.shape.slice(1)
  const per = shape.reduce((a, b) => a * b, 1)
  return { shape, data: t.data.slice(index * per, index * per + per) }
}

export function fmtNum(v: number): string {
  if (Number.isInteger(v)) return String(v)
  const a = Math.abs(v)
  return a >= 0.01 && a < 1e4 ? v.toFixed(2) : v.toExponential(1)
}

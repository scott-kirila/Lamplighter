// A tensor from /api/run/preview: a batched {shape:[n, …], data} pair (flat,
// row-major) plus a truncation flag for exotic huge per-sample sizes.
export interface TensorPayload {
  shape: number[]
  data: number[]
  truncated?: boolean
}

export type TensorKind = 'image' | 'image-grid' | 'bars' | 'scalar'

// The *per-sample* shape (batch dim already stripped) → how to draw it. Shape
// alone, no task knowledge. The one general rule for arbitrary rank: the last
// two dims are a 2-D field, everything in front is a stack of them —
//   HxW / 1xHxW / 3xHxW → a single (gray/RGB) image,
//   any higher rank / many channels (CxHxW with C>3, TxCxHxW, DxHxW) → a grid
//     of images (a feature map is a stack of images; a video, a stack of frames),
//   a vector → bars, a lone value → scalar.
// It never fires when the trailing two dims aren't spatial (both ≥2), so a
// genuine vector / sequence stays bars rather than being tiled into a fake image.
export function tensorKind(sampleShape: number[]): TensorKind {
  const numel = sampleShape.reduce((a, b) => a * b, 1)
  if (numel <= 1) return 'scalar'
  const r = sampleShape.length
  const spatial = r >= 2 && sampleShape[r - 1] >= 2 && sampleShape[r - 2] >= 2
  if (!spatial) return 'bars'
  if (r === 2) return 'image'
  if (r === 3 && (sampleShape[0] === 1 || sampleShape[0] === 3)) return 'image'
  return 'image-grid'
}

// The side length if a per-sample shape is a perfect-square vector — a
// *plausibly* flattened square (grayscale) image — else null. Used only to
// OFFER an opt-in image view; never to auto-reshape (a real F-dim vector whose
// F happens to be square must not be mangled into an image).
export function squareSide(sampleShape: number[]): number | null {
  if (sampleShape.length !== 1) return null
  const side = Math.round(Math.sqrt(sampleShape[0]))
  return side >= 2 && side * side === sampleShape[0] ? side : null
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

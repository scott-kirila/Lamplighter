import { useEffect, useRef } from 'react'
import { fmtNum, sampleAt, squareSide, tensorKind, type TensorPayload } from '../lib/tensor'

// Render one example of a batched preview tensor, chosen by its shape — an image
// (canvas), a bar chart (vector), or a number (scalar). The single primitive the
// whole "see what it learned" view composes from; it knows nothing about the task.
// `squareAsImage` (opt-in) reshapes a perfect-square vector into a grayscale
// image — recovers a flattened MNIST-style input without ever forcing it.
export function TensorView({
  tensor,
  index,
  size = 52,
  squareAsImage = false,
}: {
  tensor: TensorPayload
  index: number
  size?: number
  squareAsImage?: boolean
}) {
  const { shape, data } = sampleAt(tensor, index)
  let kind = tensorKind(shape)
  let renderShape = shape
  if (squareAsImage && kind === 'bars') {
    const side = squareSide(shape)
    if (side) {
      kind = 'image'
      renderShape = [1, side, side]
    }
  }
  if (kind === 'scalar') {
    return <span style={{ fontFamily: 'monospace', fontSize: 13, color: 'var(--text-2)' }}>{fmtNum(data[0] ?? 0)}</span>
  }
  if (kind === 'image') return <ImageTensor shape={renderShape} data={data} size={size} />
  if (kind === 'image-grid') return <ImageGrid shape={renderShape} data={data} size={size} />
  return <Bars values={data} size={size} />
}

// Higher-rank tensors: the trailing two dims are a 2-D field, everything before
// is a stack — a feature map's channels, a video's frames — so tile them as small
// grayscale images (capped, with a "+N" for the rest). Each tile normalizes on
// its own, so a per-channel contrast is visible.
function ImageGrid({ shape, data, size }: { shape: number[]; data: number[]; size: number }) {
  const h = shape[shape.length - 2]
  const w = shape[shape.length - 1]
  const hw = h * w
  const total = shape.slice(0, -2).reduce((a, b) => a * b, 1)
  const cap = 16
  const count = Math.min(total, cap)
  const tile = Math.max(16, Math.round(size / 2))
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 2, maxWidth: tile * 4 + 8, alignItems: 'flex-start' }}>
      {Array.from({ length: count }).map((_, i) => (
        <ImageTensor key={i} shape={[1, h, w]} data={data.slice(i * hw, i * hw + hw)} size={tile} />
      ))}
      {total > cap && (
        <span style={{ alignSelf: 'center', color: 'var(--text-7)', fontSize: 9 }}>+{total - cap}</span>
      )}
    </div>
  )
}

function ImageTensor({ shape, data, size }: { shape: number[]; data: number[]; size: number }) {
  const ref = useRef<HTMLCanvasElement>(null)
  const [c, h, w] = shape.length === 3 ? shape : [1, shape[0], shape[1]]
  useEffect(() => {
    const canvas = ref.current
    const ctx = canvas?.getContext('2d')
    if (!canvas || !ctx) return
    canvas.width = w
    canvas.height = h
    // Min-max normalize per sample, so any input scaling (raw, [0,1], standardized)
    // displays sensibly.
    const min = Math.min(...data)
    const span = Math.max(...data) - min || 1
    const norm = (v: number) => Math.round(((v - min) / span) * 255)
    const img = ctx.createImageData(w, h)
    const hw = h * w
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const p = (y * w + x) * 4
        if (c === 3) {
          // channel-major CHW → RGBA
          img.data[p] = norm(data[y * w + x])
          img.data[p + 1] = norm(data[hw + y * w + x])
          img.data[p + 2] = norm(data[2 * hw + y * w + x])
        } else {
          const g = norm(data[y * w + x])
          img.data[p] = img.data[p + 1] = img.data[p + 2] = g
        }
        img.data[p + 3] = 255
      }
    }
    ctx.putImageData(img, 0, 0)
  }, [data, w, h, c])
  return (
    <canvas
      ref={ref}
      style={{
        width: size,
        height: (size * h) / w,
        imageRendering: 'pixelated', // crisp nearest-neighbour upscale of the tiny image
        borderRadius: 3,
        background: 'var(--field)',
        display: 'block',
      }}
    />
  )
}

function Bars({ values, size }: { values: number[]; size: number }) {
  // Keep the DOM sane for large vectors by uniform down-sampling to ~64 bars.
  const stride = Math.max(1, Math.ceil(values.length / 64))
  const shown = stride > 1 ? values.filter((_, i) => i % stride === 0) : values
  const min = Math.min(...shown)
  const max = Math.max(...shown)
  const span = max - min || 1
  const peak = shown.indexOf(max) // the "answer" (tallest bar) — surfaced, not asserted
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'flex-end',
        gap: shown.length > 40 ? 0 : 1,
        width: size * 1.7,
        height: size,
        background: 'var(--field)',
        borderRadius: 3,
        padding: 2,
        boxSizing: 'border-box',
      }}
    >
      {shown.map((v, i) => (
        <div
          key={i}
          title={`${i * stride}: ${fmtNum(v)}`}
          style={{
            flex: 1,
            height: `${Math.max(3, ((v - min) / span) * 100)}%`,
            background: i === peak ? 'var(--accent)' : 'var(--text-6)',
          }}
        />
      ))}
    </div>
  )
}

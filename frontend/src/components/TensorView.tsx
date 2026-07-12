import { useEffect, useRef } from 'react'
import { fmtNum, sampleAt, tensorKind, type TensorPayload } from '../lib/tensor'

// Render one example of a batched preview tensor, chosen by its shape — an image
// (canvas), a bar chart (vector), or a number (scalar). The single primitive the
// whole "see what it learned" view composes from; it knows nothing about the task.
export function TensorView({ tensor, index, size = 52 }: { tensor: TensorPayload; index: number; size?: number }) {
  const { shape, data } = sampleAt(tensor, index)
  const kind = tensorKind(shape)
  if (kind === 'scalar') {
    return <span style={{ fontFamily: 'monospace', fontSize: 13, color: 'var(--text-2)' }}>{fmtNum(data[0] ?? 0)}</span>
  }
  if (kind === 'image') return <ImageTensor shape={shape} data={data} size={size} />
  return <Bars values={data} size={size} />
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

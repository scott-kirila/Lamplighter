import { useCallback, useEffect, useRef, useState } from 'react'
import { eyebrow } from '../styles/ui'

interface Frame {
  h: number
  w: number
  data: number[] // flat HWC uint8 RGB
}

interface Rollout {
  env_id: string
  steps: number
  total_return: number
  frames: Frame[]
  probs: number[][]
  actions: number[]
  returns: number[]
  error?: string
}

const note: React.CSSProperties = { color: 'var(--text-6)', fontSize: 12, padding: '8px 0' }

// Paint one rgb_array frame straight into a canvas — the frame is already true
// uint8 RGB (no normalization, unlike the grayscale ImageTensor), stride-
// downscaled at capture, so pixelated upscaling keeps the line art crisp.
function FrameCanvas({ frame, width }: { frame: Frame; width: number }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const canvas = ref.current
    const ctx = canvas?.getContext('2d')
    if (!canvas || !ctx) return
    canvas.width = frame.w
    canvas.height = frame.h
    const img = ctx.createImageData(frame.w, frame.h)
    for (let i = 0; i < frame.w * frame.h; i++) {
      img.data[i * 4] = frame.data[i * 3]
      img.data[i * 4 + 1] = frame.data[i * 3 + 1]
      img.data[i * 4 + 2] = frame.data[i * 3 + 2]
      img.data[i * 4 + 3] = 255
    }
    ctx.putImageData(img, 0, 0)
  }, [frame])
  return (
    <canvas
      ref={ref}
      style={{
        width,
        height: (width * frame.h) / frame.w,
        imageRendering: 'pixelated',
        borderRadius: 6,
        border: '1px solid var(--border)',
        background: 'var(--field)',
        display: 'block',
      }}
    />
  )
}

// RL's "see what it learned": roll out one episode with the run's policy and
// replay it as a scrubbable filmstrip — the frame, the policy's per-step action
// probabilities (the taken action highlighted), and the running return. The
// live/current run rolls out via /api/run/rollout; a viewed saved run rebuilds
// from its weights via /api/checkpoints/{name}/rollout (kernel untouched), so
// flipping between runs replays each. Mirrors PreviewView's run resolution.
export function RolloutView({
  shown,
  isLive,
  liveReady,
}: {
  shown: string | null
  isLive: boolean
  liveReady: boolean
}) {
  const [data, setData] = useState<Rollout | null>(null)
  const [loading, setLoading] = useState(false)
  const [idx, setIdx] = useState(0)
  const [playing, setPlaying] = useState(false)

  const fetchRollout = useCallback(async () => {
    if (isLive && !liveReady) {
      setData(null)
      return
    }
    setLoading(true)
    try {
      const url = isLive
        ? '/api/run/rollout'
        : `/api/checkpoints/${encodeURIComponent(shown!)}/rollout`
      const res = await fetch(url)
      const body = await res.json().catch(() => ({}))
      setData(res.ok ? body : { error: body.detail ?? 'rollout request failed' } as Rollout)
      setIdx(0)
      setPlaying(res.ok && !body.error)
    } catch {
      setData({ error: 'rollout request failed' } as Rollout)
    } finally {
      setLoading(false)
    }
  }, [isLive, liveReady, shown])

  useEffect(() => {
    fetchRollout()
  }, [fetchRollout])

  // Playback: advance ~12 fps, stop at the last frame (a scrub restarts it).
  const nFrames = data?.frames?.length ?? 0
  useEffect(() => {
    if (!playing || nFrames === 0) return
    const t = window.setInterval(() => {
      setIdx((i) => {
        if (i + 1 >= nFrames) {
          setPlaying(false)
          return i
        }
        return i + 1
      })
    }, 80)
    return () => window.clearInterval(t)
  }, [playing, nFrames])

  if (isLive && !liveReady && shown == null) {
    return <div style={note}>run the policy, or pick a run from the list, to replay an episode.</div>
  }
  if (loading && !data) return <div style={note}>rolling out an episode…</div>
  if (data?.error) return <div style={note}>{data.error}</div>
  if (!data?.frames?.length) return <div style={note}>no rollout to show yet.</div>

  const frame = data.frames[Math.min(idx, nFrames - 1)]
  const probs = data.probs[Math.min(idx, nFrames - 1)] ?? []
  const action = data.actions[Math.min(idx, nFrames - 1)]
  const runningReturn = data.returns[Math.min(idx, nFrames - 1)]

  return (
    <div style={{ padding: '4px 0' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 10, flexWrap: 'wrap', color: 'var(--text-6)', fontSize: 11 }}>
        <span style={{ ...eyebrow, color: 'var(--text-4)', fontSize: 10 }}>{data.env_id}</span>
        <span>
          return <span style={{ color: 'var(--accent)' }}>{data.total_return.toFixed(0)}</span> over {data.steps} steps
        </span>
        <span
          role="button"
          onClick={() => !loading && fetchRollout()}
          title="Roll out a fresh episode"
          style={{ color: 'var(--text-5)', cursor: 'pointer', marginLeft: 'auto' }}
        >
          ↻ new episode
        </span>
      </div>

      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', alignItems: 'flex-start' }}>
        <FrameCanvas frame={frame} width={340} />

        <div style={{ flex: 1, minWidth: 200 }}>
          {/* Per-step action probabilities — the taken action accented. The
              probs are a display-layer softmax over the policy's logits. */}
          <div style={{ ...eyebrow, fontSize: 10, color: 'var(--text-6)', marginBottom: 6 }}>
            action probabilities
          </div>
          {probs.map((p, a) => (
            <div key={a} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
              <span style={{ width: 58, color: a === action ? 'var(--accent)' : 'var(--text-5)', fontSize: 11 }}>
                action {a}
              </span>
              <div style={{ flex: 1, height: 9, background: 'var(--field)', border: '1px solid var(--border)', borderRadius: 3 }}>
                <div style={{ width: `${p * 100}%`, height: '100%', background: a === action ? 'var(--accent)' : 'var(--text-6)', borderRadius: 2 }} />
              </div>
              <span style={{ width: 40, textAlign: 'right', color: 'var(--text-5)', fontSize: 11 }}>{p.toFixed(2)}</span>
            </div>
          ))}
          <div style={{ marginTop: 10, color: 'var(--text-5)', fontSize: 11 }}>
            step {idx + 1} / {nFrames} · return so far {runningReturn?.toFixed(0)}
          </div>
        </div>
      </div>

      {/* Scrubber + play — the film transport. */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginTop: 12 }}>
        <button
          onClick={() => {
            if (idx + 1 >= nFrames) setIdx(0) // replay from the top
            setPlaying((p) => !p)
          }}
          style={{
            background: 'none', border: '1px solid var(--border)', borderRadius: 4,
            color: 'var(--text-3)', cursor: 'pointer', fontFamily: 'monospace', fontSize: 12, padding: '2px 10px',
          }}
        >
          {playing ? '❚❚' : '▶'}
        </button>
        <input
          type="range" min={0} max={Math.max(0, nFrames - 1)} value={idx}
          onChange={(e) => { setPlaying(false); setIdx(Number(e.target.value)) }}
          style={{ flex: 1 }}
        />
      </div>
    </div>
  )
}

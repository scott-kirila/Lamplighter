import { useState } from 'react'
import { useRunStore } from '../store/runStore'
import { TensorView } from './TensorView'
import type { TensorPayload } from '../lib/tensor'

interface PreviewResult {
  role?: string
  n?: number
  inputs?: TensorPayload[]
  outputs?: TensorPayload[]
  target?: TensorPayload | null
  error?: string
}

const note: React.CSSProperties = { color: 'var(--text-6)', fontSize: 12, padding: '4px 0' }

// "See what it learned": on demand, forward a sample of the trained model's real
// inputs and show input → output (vs target, when the data has one). Each tensor
// renders by shape via TensorView — no task logic, so a classifier, autoencoder,
// or generator all lay out through the same code. Fetches when opened (not on
// every finished run), with ↻ to resample.
export function PreviewPanel() {
  const runState = useRunStore((s) => s.runState)
  const [open, setOpen] = useState(false)
  const [data, setData] = useState<PreviewResult | null>(null)
  const [loading, setLoading] = useState(false)

  // Only meaningful once a run has produced a trained model.
  if (runState !== 'done' && runState !== 'stopped') return null

  const load = async () => {
    setLoading(true)
    try {
      const res = await fetch('/api/run/preview?n=12')
      setData(await res.json())
    } catch {
      setData({ error: 'preview request failed' })
    } finally {
      setLoading(false)
    }
  }

  const toggle = () => {
    const next = !open
    setOpen(next)
    if (next && data === null && !loading) load()
  }

  const n = data?.inputs ? data.n ?? 0 : 0

  return (
    <div style={{ borderTop: '1px solid var(--border)', background: 'var(--panel)', flexShrink: 0, fontFamily: 'monospace' }}>
      <button
        onClick={toggle}
        style={{
          width: '100%', display: 'flex', alignItems: 'center', gap: 10, background: 'none', border: 'none',
          cursor: 'pointer', padding: '8px 16px', fontFamily: 'monospace', fontSize: 11, color: 'var(--text-4)',
          textTransform: 'uppercase', letterSpacing: 1,
        }}
      >
        <span style={{ color: 'var(--text-6)' }}>{open ? '▾' : '▸'}</span>
        Model preview
        {open && data?.inputs && (
          <span
            role="button"
            onClick={(e) => {
              e.stopPropagation()
              if (!loading) load()
            }}
            title="Resample"
            style={{ marginLeft: 'auto', color: 'var(--text-5)', textTransform: 'none', letterSpacing: 0 }}
          >
            ↻
          </span>
        )}
      </button>

      {open && (
        <div style={{ padding: '0 16px 14px', maxHeight: 280, overflowY: 'auto' }}>
          {loading ? (
            <div style={note}>sampling…</div>
          ) : data?.error ? (
            <div style={note}>{data.error}</div>
          ) : data?.inputs ? (
            <>
              <div style={{ color: 'var(--text-7)', fontSize: 10, marginBottom: 8 }}>
                input → output{data.target ? ' vs target' : ''} · {n} samples
                {data.role ? ` · ${data.role}` : ''}
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
                {Array.from({ length: n }).map((_, i) => (
                  <div
                    key={i}
                    style={{
                      display: 'flex', alignItems: 'center', gap: 6, padding: 6,
                      border: '1px solid var(--border)', borderRadius: 6, background: 'var(--bg)',
                    }}
                  >
                    {data.inputs!.map((t, k) => <TensorView key={`in${k}`} tensor={t} index={i} />)}
                    <span style={{ color: 'var(--text-6)' }}>→</span>
                    {data.outputs!.map((t, k) => <TensorView key={`out${k}`} tensor={t} index={i} />)}
                    {data.target && (
                      <>
                        <span style={{ color: 'var(--text-7)', fontSize: 10 }}>vs</span>
                        <TensorView tensor={data.target} index={i} />
                      </>
                    )}
                  </div>
                ))}
              </div>
            </>
          ) : null}
        </div>
      )}
    </div>
  )
}

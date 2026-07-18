import { useCallback, useEffect, useState } from 'react'
import { useRunStore } from '../store/runStore'
import { TensorView } from './TensorView'
import { squareSide, type TensorPayload } from '../lib/tensor'

interface PreviewResult {
  role?: string
  n?: number
  inputs?: TensorPayload[]
  outputs?: TensorPayload[]
  target?: TensorPayload | null
  error?: string
}

const note: React.CSSProperties = { color: 'var(--text-6)', fontSize: 12, padding: '8px 0' }

// "See what it learned" as its own Training sub-tab: forward a sample of a run's
// real inputs and show input → output (vs target, when the data has one). Each
// tensor renders by shape via TensorView — no task logic, so a classifier,
// autoencoder, or generator all lay out through the same code.
//
// It follows the SHOWN run (runStore.runName): the live/current run previews via
// /api/run/preview; a viewed-but-not-live saved run previews by name via
// /api/checkpoints/{name}/preview (rebuilt from its saved weights, kernel
// untouched) — so clicking between runs in the shared runs list re-samples each.
export function PreviewView() {
  const runState = useRunStore((s) => s.runState)
  const runName = useRunStore((s) => s.runName)
  const kernelRunName = useRunStore((s) => s.kernelRunName)
  const [data, setData] = useState<PreviewResult | null>(null)
  const [loading, setLoading] = useState(false)
  // Opt-in: reshape perfect-square vectors into images (a flattened MNIST-style
  // input), off by default so a real vector is never mangled.
  const [asImage, setAsImage] = useState(false)

  // The shown run is "live" when it's the run the kernel holds (or nothing is
  // pinned) — then the current model answers; otherwise preview it by name.
  const shown = runName
  const isLive = shown == null || shown === kernelRunName
  // The live model only exists once a run has finished; a saved run always can.
  const liveReady = runState === 'done' || runState === 'stopped'

  const fetchPreview = useCallback(async () => {
    if (isLive && !liveReady) {
      setData(null)
      return
    }
    setLoading(true)
    try {
      const url = isLive
        ? '/api/run/preview?n=12'
        : `/api/checkpoints/${encodeURIComponent(shown!)}/preview?n=12`
      const res = await fetch(url)
      const body = await res.json().catch(() => ({}))
      setData(res.ok ? body : { error: body.detail ?? 'preview request failed' })
    } catch {
      setData({ error: 'preview request failed' })
    } finally {
      setLoading(false)
    }
  }, [isLive, liveReady, shown])

  // Re-sample whenever the shown run (or the live model's state) changes, so
  // flipping between runs shows each one's outputs.
  useEffect(() => {
    fetchPreview()
  }, [fetchPreview])

  const n = data?.inputs ? data.n ?? 0 : 0
  // Offer the "as image" toggle only when a perfect-square vector is present.
  const hasSquare = [...(data?.inputs ?? []), ...(data?.outputs ?? []), data?.target].some(
    (t) => t != null && squareSide(t.shape.slice(1)) != null
  )

  const nothingToShow = isLive && !liveReady && runName == null

  return (
    <div style={{ height: '100%', overflowY: 'auto', background: 'var(--panel)', fontFamily: 'monospace', padding: '10px 16px' }}>
      <div
        style={{
          display: 'flex', alignItems: 'center', gap: 12, color: 'var(--text-7)', fontSize: 10,
          marginBottom: 10, flexWrap: 'wrap',
        }}
      >
        <span style={{ color: 'var(--text-4)', textTransform: 'uppercase', letterSpacing: 1 }}>
          Model preview
        </span>
        <span style={{ color: 'var(--text-6)' }}>{runName ?? '(current run)'}{isLive ? ' · live' : ''}</span>
        {data?.inputs && (
          <>
            <span>input → output{data.target ? ' vs target' : ''} · {n} samples{data.role ? ` · ${data.role}` : ''}</span>
            <span
              role="button"
              onClick={() => !loading && fetchPreview()}
              title="Resample"
              style={{ color: 'var(--text-5)', cursor: 'pointer' }}
            >
              ↻
            </span>
          </>
        )}
        {hasSquare && (
          <label style={{ display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer', marginLeft: 'auto' }}>
            <input type="checkbox" checked={asImage} onChange={(e) => setAsImage(e.target.checked)} />
            square vectors as images
          </label>
        )}
      </div>

      {nothingToShow ? (
        <div style={note}>run a model, or pick a run from the list, to preview its outputs.</div>
      ) : data?.inputs ? (
        <div
          style={{
            display: 'flex', flexWrap: 'wrap', gap: 10,
            // Keep the grid mounted during a resample (just dim it) so the view
            // never collapses and flashes.
            opacity: loading ? 0.4 : 1, transition: 'opacity 0.12s',
          }}
        >
          {Array.from({ length: n }).map((_, i) => (
            <div
              key={i}
              style={{
                display: 'flex', alignItems: 'center', gap: 6, padding: 6,
                border: '1px solid var(--border)', borderRadius: 6, background: 'var(--bg)',
              }}
            >
              {data.inputs!.map((t, k) => <TensorView key={`in${k}`} tensor={t} index={i} squareAsImage={asImage} />)}
              <span style={{ color: 'var(--text-6)' }}>→</span>
              {data.outputs!.map((t, k) => <TensorView key={`out${k}`} tensor={t} index={i} squareAsImage={asImage} />)}
              {data.target && (
                <>
                  <span style={{ color: 'var(--text-7)', fontSize: 10 }}>vs</span>
                  <TensorView tensor={data.target} index={i} squareAsImage={asImage} />
                </>
              )}
            </div>
          ))}
        </div>
      ) : loading ? (
        <div style={note}>sampling…</div>
      ) : data?.error ? (
        <div style={note}>{data.error}</div>
      ) : (
        <div style={note}>training in progress — preview once the run finishes.</div>
      )}
    </div>
  )
}

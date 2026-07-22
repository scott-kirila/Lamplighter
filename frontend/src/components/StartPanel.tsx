import { useState } from 'react'
import { useLoadTemplate } from '../hooks/useLoadTemplate'
import { useTemplates } from '../hooks/useTemplates'
import { border, eyebrow } from '../styles/ui'
import type { NodeDef } from '../types/graph'

/**
 * What an empty canvas offers instead of nothing.
 *
 * A fresh session opens on a bare Input and Output with no wire and no data,
 * and the only routes out — the Templates menu, dragging from the palette,
 * registering tensors in the notebook — are all things you have to already know
 * about. So the first screen of a brand-new install was a dot grid and two
 * disconnected boxes.
 *
 * The offer is one click to something that trains, because the fastest way to
 * understand this app is to watch a loss curve come out of it. Everything else
 * here is a link to the next-most-likely intent, not a tour.
 */
export function StartPanel({ registry }: { registry: Record<string, NodeDef> }) {
  const { data: templates } = useTemplates(true)
  const loadTemplate = useLoadTemplate(registry)
  const [busy, setBusy] = useState<string | null>(null)
  const [failed, setFailed] = useState(false)
  const [browsing, setBrowsing] = useState(false)

  const run = async (name: string) => {
    setBusy(name)
    setFailed(!(await loadTemplate(name)))
    setBusy(null)
  }

  // The zero-setup one leads; the rest are the "I know what I want" list.
  const zeroSetup = (templates ?? []).find((t) => t.name === 'mnist')
  const rest = (templates ?? []).filter((t) => t.name !== 'mnist')

  return (
    <div
      style={{
        position: 'absolute', inset: 0, display: 'flex',
        alignItems: 'center', justifyContent: 'center',
        pointerEvents: 'none', padding: 24,
      }}
    >
      <div
        style={{
          pointerEvents: 'auto', background: 'var(--panel)', border,
          borderRadius: 10, padding: '22px 26px', maxWidth: 440,
          boxShadow: 'var(--shadow-md)',
        }}
      >
        <div style={{ ...eyebrow, fontSize: 10, color: 'var(--text-4)', marginBottom: 10 }}>
          Start here
        </div>

        {zeroSetup && (
          <>
            <button
              onClick={() => run(zeroSetup.name)}
              disabled={busy !== null}
              style={{
                width: '100%', textAlign: 'left', background: 'var(--accent-fill)',
                color: 'var(--text-on-accent)', border: 'none', borderRadius: 6,
                padding: '10px 14px', fontSize: 13, fontWeight: 600,
                cursor: busy ? 'default' : 'pointer', opacity: busy ? 0.6 : 1,
              }}
            >
              {busy === zeroSetup.name ? 'Loading…' : `▶ ${zeroSetup.label}`}
            </button>
            <div style={{ color: 'var(--text-4)', fontSize: 11.5, margin: '8px 0 18px', lineHeight: 1.55 }}>
              {zeroSetup.description}
            </div>
          </>
        )}

        <div style={{ color: 'var(--text-5)', fontSize: 11.5, lineHeight: 1.6 }}>
          Or drag a node from the palette to build your own, and register your
          tensors in the notebook with{' '}
          <span style={{ color: 'var(--accent)' }}>sess.data(X=X, y=y)</span>.
        </div>

        {rest.length > 0 && (
          <>
            <button
              onClick={() => setBrowsing((v) => !v)}
              aria-expanded={browsing}
              style={{
                background: 'none', border: 'none', color: 'var(--text-4)',
                fontSize: 11.5, cursor: 'pointer', padding: '10px 0 0', display: 'block',
              }}
            >
              {browsing ? '▾' : '▸'} {rest.length} more templates
            </button>
            {browsing && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 2, paddingTop: 6 }}>
                {rest.map((t) => (
                  <button
                    key={t.name}
                    onClick={() => run(t.name)}
                    disabled={busy !== null}
                    style={{
                      background: 'none', border: 'none', borderRadius: 4,
                      color: 'var(--text-3)', fontSize: 11.5, cursor: 'pointer',
                      padding: '4px 6px', textAlign: 'left',
                    }}
                  >
                    {busy === t.name ? 'Loading…' : t.label}
                  </button>
                ))}
              </div>
            )}
          </>
        )}

        {failed && (
          <div style={{ color: 'var(--error)', fontSize: 11.5, marginTop: 10 }}>
            ✗ couldn't reach the kernel — is the session still running?
          </div>
        )}
      </div>
    </div>
  )
}

import { createPortal } from 'react-dom'
import { chip } from '../styles/ui'

const modalBtn = (primary: boolean): React.CSSProperties => ({
  background: 'none',
  border: `1px solid ${primary ? 'var(--accent)' : 'var(--border)'}`,
  borderRadius: 4,
  color: primary ? 'var(--accent)' : 'var(--text-3)',
  cursor: 'pointer',
  fontFamily: 'monospace',
  fontSize: 11,
  padding: '3px 10px',
})

// Starting a run replaces the current model — warn (raised from the Preview tab)
// when that would drop the live run's unsaved weights. Portaled so nothing in the
// pane reflows. `onConfirm(true)` saves the weights first; `onConfirm(false)`
// discards them.
export function DiscardWeightsModal({
  kernelRunName,
  onCancel,
  onConfirm,
}: {
  kernelRunName: string | null
  onCancel: () => void
  onConfirm: (save: boolean) => void
}) {
  return createPortal(
    <div
      onClick={onCancel}
      style={{
        position: 'fixed', inset: 0, zIndex: 1000, background: 'rgba(0, 0, 0, 0.45)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: 'var(--panel)', border: '1px solid var(--border)', borderRadius: 8,
          padding: 16, width: 'min(380px, calc(100vw - 32px))', boxShadow: '0 8px 30px rgba(0, 0, 0, 0.35)',
          fontFamily: 'monospace', fontSize: 12, display: 'flex', flexDirection: 'column', gap: 14,
        }}
      >
        <span style={{ color: 'var(--text)', lineHeight: 1.6 }}>
          <code style={chip}>{kernelRunName}</code> is the current model and its weights aren't
          saved. Starting a new run will discard them.
        </span>
        <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end', flexWrap: 'wrap' }}>
          <button onClick={onCancel} style={modalBtn(false)}>
            cancel
          </button>
          <button onClick={() => onConfirm(false)} style={modalBtn(false)}>
            discard &amp; run
          </button>
          <button onClick={() => onConfirm(true)} style={modalBtn(true)}>
            save weights &amp; run
          </button>
        </div>
      </div>
    </div>,
    document.body
  )
}

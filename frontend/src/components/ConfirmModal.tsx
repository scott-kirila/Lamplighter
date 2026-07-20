import { useEffect } from 'react'
import { createPortal } from 'react-dom'

export interface ModalAction {
  label: string
  onClick: () => void
  // The affirmative action renders accent-styled; the rest are ghost buttons.
  primary?: boolean
}

const overlay: React.CSSProperties = {
  position: 'fixed', inset: 0, zIndex: 1000, background: 'rgba(0, 0, 0, 0.45)',
  display: 'flex', alignItems: 'center', justifyContent: 'center',
  animation: 'lamplighter-fade 160ms ease',
}

const modalBtn = (primary: boolean): React.CSSProperties => ({
  background: 'none',
  border: `1px solid ${primary ? 'var(--accent)' : 'var(--border)'}`,
  borderRadius: 4,
  color: primary ? 'var(--accent)' : 'var(--text-3)',
  cursor: 'pointer',
  fontSize: 11,
  padding: '3px 10px',
})

// The one confirmation modal for the whole app: a portaled overlay + panel with
// a message (children) and a right-aligned button row — a leading Cancel plus
// the affirmative `actions`. Clicking the backdrop or pressing Escape cancels.
// Every confirm flow (new project, discard-weights-before-run/restore) routes
// through this, so they look and behave identically — nothing in the pane
// reflows, since it's portaled to <body>.
export function ConfirmModal({
  children,
  actions,
  onCancel,
  cancelLabel = 'cancel',
  width = 380,
}: {
  children: React.ReactNode
  actions: ModalAction[]
  onCancel: () => void
  cancelLabel?: string
  width?: number
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onCancel()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onCancel])

  return createPortal(
    <div onClick={onCancel} style={overlay}>
      <div
        role="dialog"
        aria-modal
        onClick={(e) => e.stopPropagation()}
        style={{
          background: 'var(--panel)', border: '1px solid var(--border)', borderRadius: 8,
          padding: 16, width: `min(${width}px, calc(100vw - 32px))`,
          boxShadow: 'var(--shadow-lg)', fontSize: 12,
          display: 'flex', flexDirection: 'column', gap: 14,
          animation: 'lamplighter-enter 160ms ease',
        }}
      >
        <div style={{ color: 'var(--text)', lineHeight: 1.6 }}>{children}</div>
        <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end', flexWrap: 'wrap' }}>
          <button onClick={onCancel} style={modalBtn(false)}>
            {cancelLabel}
          </button>
          {actions.map((a) => (
            <button key={a.label} onClick={a.onClick} style={modalBtn(!!a.primary)}>
              {a.label}
            </button>
          ))}
        </div>
      </div>
    </div>,
    document.body
  )
}

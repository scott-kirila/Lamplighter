import { useRef, useState } from 'react'

interface CodePanelProps {
  code: string | null
  onClose: () => void
  title?: string
}

const MIN_HEIGHT = 120
const HEIGHT_KEY = 'lamplighter-codepanel-height'

function storedHeight(): number {
  const n = Number(localStorage.getItem(HEIGHT_KEY))
  return Number.isFinite(n) && n >= MIN_HEIGHT ? n : 260
}

export function CodePanel({ code, onClose, title = 'Generated code' }: CodePanelProps) {
  const [copied, setCopied] = useState(false)
  // Height is per-instance state but persisted, so the Model and Training
  // mounts share one remembered size (like the training split's layout).
  const [height, setHeight] = useState(storedHeight)
  const heightRef = useRef(height)
  const dragFrom = useRef<{ y: number; h: number } | null>(null)

  const handleCopy = async () => {
    if (!code) return
    try {
      await navigator.clipboard.writeText(code)
      setCopied(true)
      setTimeout(() => setCopied(false), 1200)
    } catch {
      // clipboard unavailable (insecure context) — silently ignore
    }
  }

  const onDragStart = (e: React.PointerEvent) => {
    dragFrom.current = { y: e.clientY, h: heightRef.current }
    e.currentTarget.setPointerCapture(e.pointerId)
  }
  const onDragMove = (e: React.PointerEvent) => {
    if (!dragFrom.current) return
    const next = Math.min(
      Math.round(window.innerHeight * 0.8),
      Math.max(MIN_HEIGHT, dragFrom.current.h + (dragFrom.current.y - e.clientY)),
    )
    heightRef.current = next
    setHeight(next)
  }
  const onDragEnd = (e: React.PointerEvent) => {
    if (!dragFrom.current) return
    dragFrom.current = null
    e.currentTarget.releasePointerCapture(e.pointerId)
    localStorage.setItem(HEIGHT_KEY, String(heightRef.current))
  }

  return (
    <div
      style={{
        height,
        background: 'var(--bg)',
        borderTop: '1px solid var(--border)',
        display: 'flex',
        flexDirection: 'column',
        flexShrink: 0,
        position: 'relative',
      }}
    >
      {/* Drag handle straddling the top border — pulls the panel taller/shorter. */}
      <div
        onPointerDown={onDragStart}
        onPointerMove={onDragMove}
        onPointerUp={onDragEnd}
        style={{
          position: 'absolute',
          top: -3,
          left: 0,
          right: 0,
          height: 7,
          cursor: 'row-resize',
          touchAction: 'none',
          zIndex: 1,
        }}
      />
      {/* Panel header */}
      <div
        style={{
          height: 32,
          background: 'var(--panel)',
          borderBottom: '1px solid var(--border)',
          display: 'flex',
          alignItems: 'center',
          padding: '0 12px',
          gap: 10,
          flexShrink: 0,
        }}
      >
        <span
          style={{
            fontSize: 11,
            color: 'var(--text-4)',
            textTransform: 'uppercase',
            letterSpacing: 1,
          }}
        >
          {title}
        </span>
        <button
          onClick={handleCopy}
          disabled={!code}
          style={{
            marginLeft: 'auto',
            background: 'none',
            border: '1px solid var(--border)',
            borderRadius: 4,
            color: code ? 'var(--text-3)' : 'var(--text-7)',
            cursor: code ? 'pointer' : 'default',
            fontSize: 11,
            padding: '3px 10px',
          }}
        >
          {copied ? 'Copied' : 'Copy'}
        </button>
        <button
          onClick={onClose}
          title="Hide code panel"
          style={{
            background: 'none',
            border: 'none',
            color: 'var(--text-5)',
            cursor: 'pointer',
            fontSize: 16,
            lineHeight: 1,
            padding: 0,
          }}
        >
          ×
        </button>
      </div>

      {/* Code body */}
      {code ? (
        <pre
          style={{
            margin: 0,
            padding: '12px 16px',
            overflow: 'auto',
            flex: 1,
            fontSize: 12,
            lineHeight: 1.5,
            color: 'var(--text)',
            whiteSpace: 'pre',
          }}
        >
          {code}
        </pre>
      ) : (
        <div
          style={{
            flex: 1,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: 12,
            color: 'var(--text-6)',
            padding: 16,
            textAlign: 'center',
          }}
        >
          Resolve the graph's errors to generate the model.
        </div>
      )}
    </div>
  )
}

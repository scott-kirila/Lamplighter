import { useState } from 'react'

interface CodePanelProps {
  code: string | null
  onClose: () => void
}

export function CodePanel({ code, onClose }: CodePanelProps) {
  const [copied, setCopied] = useState(false)

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

  return (
    <div
      style={{
        height: 260,
        background: 'var(--bg)',
        borderTop: '1px solid var(--border)',
        display: 'flex',
        flexDirection: 'column',
        flexShrink: 0,
      }}
    >
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
            fontFamily: 'monospace',
            fontSize: 11,
            color: 'var(--text-4)',
            textTransform: 'uppercase',
            letterSpacing: 1,
          }}
        >
          Generated code
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
            fontFamily: 'monospace',
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
            fontFamily: 'monospace',
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
            fontFamily: 'monospace',
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

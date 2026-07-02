import { useEffect, useRef, useState } from 'react'
import { useGraphStore } from '../store/graphStore'
import { useTrainingParams } from '../hooks/useTrainingParams'
import { formatEpochLine } from '../lib/formatEpochLine'
import { paramVisible } from '../lib/paramVisible'
import { ParamControl } from './Inspector'

const RUN_STATE_COLOR: Record<string, string> = {
  running: 'var(--warn)',
  done: 'var(--accent)',
  stopped: 'var(--text-4)',
  failed: 'var(--error)',
}

export function TrainingTab() {
  const { data: params } = useTrainingParams()
  const training = useGraphStore((s) => s.training)
  const setTrainingParam = useGraphStore((s) => s.setTrainingParam)
  const nodes = useGraphStore((s) => s.nodes)
  const shapes = useGraphStore((s) => s.shapes)
  const toDomainGraph = useGraphStore((s) => s.toDomainGraph)
  const runState = useGraphStore((s) => s.runState)
  const runEpochs = useGraphStore((s) => s.runEpochs)
  const runError = useGraphStore((s) => s.runError)
  const setRunStatus = useGraphStore((s) => s.setRunStatus)

  // Output-shape readout — context for choosing a loss without seeing the canvas.
  const outputNode = nodes.find((n) => n.data.nodeType === 'Output')
  const outShape = outputNode ? shapes[outputNode.id] : undefined

  // Effective config (stored over defaults) for evaluating show_if — so gated
  // params (batch_size/val_split under data=dataloader) hide correctly.
  const defaults = Object.fromEntries((params ?? []).map((p) => [p.name, p.default]))

  // Start/stop the in-kernel run. Progress/state stream back over the WS
  // (run_status / run_epoch → store); only pre-flight rejections surface here.
  const startRun = async () => {
    try {
      const res = await fetch('/api/run/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(toDomainGraph()),
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        setRunStatus('failed', body.detail ?? 'could not start the run')
      }
    } catch {
      setRunStatus('failed', 'backend unreachable')
    }
  }
  const stopRun = () => fetch('/api/run/stop', { method: 'POST' }).catch(() => {})

  // Keep the newest epoch line visible as they stream in.
  const epochsEndRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    epochsEndRef.current?.scrollIntoView({ block: 'nearest' })
  }, [runEpochs.length])

  // Generated train() preview. Refetched (debounced) after a config change, by
  // which time the change has synced to the backend via the validation socket.
  const [code, setCode] = useState<string | null>(null)
  const trainingKey = JSON.stringify(training)
  useEffect(() => {
    let cancelled = false
    const t = window.setTimeout(async () => {
      try {
        const res = await fetch('/api/training/code')
        if (res.ok && !cancelled) setCode((await res.json()).code)
      } catch {
        /* backend hiccup — leave the last preview */
      }
    }, 400)
    return () => {
      cancelled = true
      window.clearTimeout(t)
    }
  }, [trainingKey])

  return (
    <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
      {/* Config form */}
      <div
        style={{
          width: 320,
          background: 'var(--panel)',
          borderRight: '1px solid var(--border)',
          padding: 20,
          overflowY: 'auto',
          fontFamily: 'monospace',
          flexShrink: 0,
        }}
      >
        <div style={{ color: 'var(--text)', fontWeight: 700, fontSize: 14, marginBottom: 4 }}>
          Training
        </div>
        <div style={{ color: 'var(--text-6)', fontSize: 11, marginBottom: 16 }}>
          model output:{' '}
          <span style={{ color: 'var(--accent)' }}>
            {outShape ? `[${outShape.join(', ')}]` : '—'}
          </span>
        </div>

        {(params ?? [])
          .filter((param) => paramVisible(param, { ...defaults, ...training }))
          .map((param) => (
          <div key={param.name} style={{ marginBottom: 14 }}>
            <label style={{ display: 'block', color: 'var(--text-5)', fontSize: 11, marginBottom: 4 }}>
              {param.label}
            </label>
            <ParamControl
              param={param}
              value={training[param.name]}
              nodeColor="var(--accent)"
              onChange={(next) => setTrainingParam(param.name, next)}
            />
          </div>
        ))}
      </div>

      {/* Generated train() preview */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', background: 'var(--bg)' }}>
        <div
          style={{
            height: 36,
            background: 'var(--panel)',
            borderBottom: '1px solid var(--border)',
            display: 'flex',
            alignItems: 'center',
            gap: 10,
            padding: '0 16px',
            fontFamily: 'monospace',
            fontSize: 11,
            color: 'var(--text-4)',
            flexShrink: 0,
          }}
        >
          <span style={{ textTransform: 'uppercase', letterSpacing: 1 }}>Generated train()</span>
          <span style={{ marginLeft: 'auto', color: RUN_STATE_COLOR[runState] ?? 'var(--text-6)' }}>
            {runState === 'idle' ? '' : runState}
          </span>
          {runState === 'running' ? (
            <button
              onClick={stopRun}
              style={{
                background: 'none', border: '1px solid var(--border)', borderRadius: 5,
                color: 'var(--error)', cursor: 'pointer', fontFamily: 'monospace',
                fontSize: 12, fontWeight: 600, padding: '3px 14px',
              }}
            >
              ■ Stop
            </button>
          ) : (
            <button
              onClick={startRun}
              title="Train in the notebook kernel using the picked data — runs exactly this code"
              style={{
                background: 'var(--accent)', border: 'none', borderRadius: 5,
                color: 'var(--text-on-accent)', cursor: 'pointer', fontFamily: 'monospace',
                fontSize: 12, fontWeight: 600, padding: '4px 16px',
              }}
            >
              ▶ Run
            </button>
          )}
        </div>
        <pre
          style={{
            margin: 0,
            padding: '16px 20px',
            overflow: 'auto',
            flex: 1,
            fontFamily: 'monospace',
            fontSize: 12,
            lineHeight: 1.5,
            color: 'var(--text)',
            whiteSpace: 'pre',
          }}
        >
          {code ?? ''}
        </pre>

        {/* Streamed run output — epoch lines + errors from the in-kernel run. */}
        {(runEpochs.length > 0 || runError) && (
          <div
            style={{
              borderTop: '1px solid var(--border)',
              background: 'var(--panel)',
              maxHeight: 180,
              overflowY: 'auto',
              padding: '10px 20px',
              fontFamily: 'monospace',
              fontSize: 12,
              lineHeight: 1.6,
              flexShrink: 0,
            }}
          >
            {runEpochs.map((e) => (
              <div key={e.epoch} style={{ color: 'var(--text-3)', whiteSpace: 'pre' }}>
                {formatEpochLine(e)}
              </div>
            ))}
            {runError && <div style={{ color: 'var(--error)' }}>✗ {runError}</div>}
            <div ref={epochsEndRef} />
          </div>
        )}
      </div>
    </div>
  )
}

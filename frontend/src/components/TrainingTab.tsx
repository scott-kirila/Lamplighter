import { useEffect, useRef } from 'react'
import { useGraphStore } from '../store/graphStore'
import { useTrainingParams } from '../hooks/useTrainingParams'
import { formatEpochLine } from '../lib/formatEpochLine'
import { formatShape } from '../lib/formatShape'
import { paramVisible } from '../lib/paramVisible'
import { ParamControl } from './Inspector'
import { RunCharts } from './RunCharts'

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
  const paramCounts = useGraphStore((s) => s.paramCounts)
  const toDomainGraph = useGraphStore((s) => s.toDomainGraph)
  const runState = useGraphStore((s) => s.runState)
  const runEpochs = useGraphStore((s) => s.runEpochs)
  const runError = useGraphStore((s) => s.runError)
  const setRunStatus = useGraphStore((s) => s.setRunStatus)

  // Output-shape + size readout — context for choosing a loss without the canvas.
  const outputNode = nodes.find((n) => n.data.nodeType === 'Output')
  const outShape = outputNode ? shapes[outputNode.id] : undefined
  const totalParams = Object.values(paramCounts).reduce((a, b) => a + b.count, 0)

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
        <div style={{ color: 'var(--text-6)', fontSize: 11, marginBottom: 4 }}>
          model output:{' '}
          <span style={{ color: 'var(--accent)' }}>
            {outShape ? `[${formatShape(outShape, ', ')}]` : '—'}
          </span>
        </div>
        <div style={{ color: 'var(--text-6)', fontSize: 11, marginBottom: 16 }}>
          parameters:{' '}
          <span style={{ color: 'var(--accent)' }}>
            {totalParams > 0 ? totalParams.toLocaleString('en-US') : '—'}
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

      {/* Run dashboard — live charts + epoch log. The generated train() opens
          via the titlebar's Show code button (a CodePanel, like the Model tab). */}
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
          <span style={{ textTransform: 'uppercase', letterSpacing: 1 }}>Training run</span>
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
        {runEpochs.length === 0 && !runError ? (
          // Nothing streamed yet — point at the workflow instead of blank space.
          <div
            style={{
              flex: 1,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontFamily: 'monospace',
              fontSize: 12,
              color: 'var(--text-6)',
              padding: 24,
              textAlign: 'center',
              lineHeight: 1.8,
            }}
          >
            {runState === 'running'
              ? 'starting…'
              : 'Pick data in the Data tab, set the loop here, then press ▶ Run — live metrics stream in. (Show code previews the exact train() that runs.)'}
          </div>
        ) : (
          <div
            style={{
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
              minHeight: 0,
              padding: '14px 20px',
              fontFamily: 'monospace',
              fontSize: 12,
              lineHeight: 1.6,
            }}
          >
            <RunCharts epochs={runEpochs} height={200} />
            <div style={{ flex: 1, minHeight: 0, overflowY: 'auto' }}>
              {runEpochs.map((e) => (
                <div key={e.epoch} style={{ color: 'var(--text-3)', whiteSpace: 'pre' }}>
                  {formatEpochLine(e)}
                </div>
              ))}
              {runError && <div style={{ color: 'var(--error)' }}>✗ {runError}</div>}
              <div ref={epochsEndRef} />
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

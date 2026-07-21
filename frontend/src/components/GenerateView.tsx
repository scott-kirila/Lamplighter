import { useState } from 'react'
import { field } from '../styles/ui'

interface Sample {
  prompt: string
  text: string
  completion: string
  temperature: number
  vocab_size: number
  error?: string
}

const note: React.CSSProperties = { color: 'var(--text-6)', fontSize: 12, padding: '8px 0' }

// A language model's preview: what it WRITES. A loss curve says how surprised
// the model is; only a sample says whether it learned the shape of the text.
// The prompt is rendered dimmed and the continuation bright, so it's always
// clear which part the model produced.
export function GenerateView({
  shown,
  isLive,
  liveReady,
}: {
  shown: string | null
  isLive: boolean
  liveReady: boolean
}) {
  const [prompt, setPrompt] = useState('')
  const [temperature, setTemperature] = useState(0.8)
  const [maxTokens, setMaxTokens] = useState(200)
  const [sample, setSample] = useState<Sample | null>(null)
  const [busy, setBusy] = useState(false)

  const ready = !isLive || liveReady
  const generate = async () => {
    setBusy(true)
    try {
      const res = await fetch('/api/run/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt,
          temperature,
          max_new_tokens: maxTokens,
          // A stored run samples from its own saved weights; the kernel's run
          // needs no name.
          name: isLive ? null : shown,
        }),
      })
      const body = await res.json().catch(() => ({}))
      setSample(res.ok ? body : { error: body.detail ?? 'generation failed' } as Sample)
    } catch {
      setSample({ error: 'backend unreachable' } as Sample)
    } finally {
      setBusy(false)
    }
  }

  if (!ready) {
    return <div style={note}>Train a run, then sample from it here.</div>
  }

  return (
    <div>
      <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap', marginBottom: 12 }}>
        <input
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !busy) generate()
          }}
          placeholder="a prompt to continue (or leave empty)"
          data-tip="The model continues this text. Empty starts from a random token."
          style={{ ...field, flex: 1, minWidth: 220 }}
        />
        <label style={{ display: 'flex', alignItems: 'center', gap: 6, color: 'var(--text-5)', fontSize: 11 }}>
          temperature
          <input
            type="number" step={0.1} min={0} max={2} value={temperature}
            onChange={(e) => setTemperature(Number(e.target.value))}
            data-tip="Low sharpens toward the likeliest token (repetitive); high flattens the odds (wilder)"
            style={{ ...field, width: 64, padding: '3px 6px' }}
          />
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: 6, color: 'var(--text-5)', fontSize: 11 }}>
          tokens
          <input
            type="number" step={50} min={1} max={2000} value={maxTokens}
            onChange={(e) => setMaxTokens(Number(e.target.value))}
            style={{ ...field, width: 72, padding: '3px 6px' }}
          />
        </label>
        <button
          onClick={generate}
          disabled={busy}
          style={{
            background: 'var(--accent)', border: 'none', borderRadius: 5,
            color: 'var(--text-on-accent)', fontSize: 12, fontWeight: 600,
            padding: '4px 16px', cursor: busy ? 'default' : 'pointer', opacity: busy ? 0.6 : 1,
          }}
        >
          {busy ? 'writing…' : 'Generate'}
        </button>
      </div>

      {sample?.error && <div style={{ ...note, color: 'var(--error)' }}>{sample.error}</div>}

      {sample && !sample.error && (
        <>
          <pre
            style={{
              margin: 0, padding: 12, background: 'var(--surface)',
              border: '1px solid var(--border)', borderRadius: 6,
              fontSize: 12, lineHeight: 1.6, whiteSpace: 'pre-wrap', wordBreak: 'break-word',
              color: 'var(--text-2)', maxHeight: '60vh', overflowY: 'auto',
            }}
          >
            <span style={{ color: 'var(--text-6)' }}>{sample.prompt}</span>
            {sample.completion}
          </pre>
          <div style={{ ...note, fontSize: 11 }}>
            {sample.completion.length} characters at temperature {sample.temperature} · vocabulary
            of {sample.vocab_size} · the dimmed text is your prompt
          </div>
        </>
      )}

      {!sample && (
        <div style={note}>
          Sample a continuation to see what it learned — a loss curve says how surprised the model
          is, not what it writes.
        </div>
      )}
    </div>
  )
}

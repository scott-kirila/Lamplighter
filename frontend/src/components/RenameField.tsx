import { useEffect, useRef, useState } from 'react'

// The rename field: renames LIVE while typing, but a transiently-empty box
// never commits (the sidebar's rename guards empty; this path must too — a
// nameless model/data node would flow into role dropdowns, run attribution,
// and the generated class name). Blur restores the last real name if left
// empty, and commits trimmed. External renames (the sidebar, a remote tab)
// re-seed the draft — the emitted-ref pattern the param editors use.
export function RenameField({
  name,
  onRename,
  style,
}: {
  name: string
  onRename: (v: string) => void
  style?: React.CSSProperties
}) {
  const [draft, setDraft] = useState(name)
  const emitted = useRef(name)
  useEffect(() => {
    if (name !== emitted.current) {
      setDraft(name)
      emitted.current = name
    }
  }, [name])
  return (
    <input
      value={draft}
      onChange={(e) => {
        setDraft(e.target.value)
        const v = e.target.value.trim()
        if (v) {
          emitted.current = v
          onRename(v)
        }
      }}
      onBlur={() => {
        const v = draft.trim()
        if (!v) setDraft(emitted.current)
        else if (v !== draft) setDraft(v) // shed stray whitespace
      }}
      style={{
        width: '100%', background: 'var(--field)', color: 'var(--text)', border: '1px solid var(--border)',
        borderRadius: 5, padding: '6px 8px', fontSize: 14, fontWeight: 700,
        ...style,
      }}
    />
  )
}

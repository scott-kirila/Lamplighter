// The overview canvases' connection dots — one look for model and data nodes.
export const handleStyle = {
  background: 'var(--text-6)',
  width: 11,
  height: 11,
  border: '2px solid var(--border)',
} as const

// One data-node kind → colour mapping, shared by the overview node header and
// the sidebar rows — a green env NODE must have a green sidebar ROW.
export const dataKindColor = (kind: string): string =>
  kind === 'noise' ? 'var(--warn)' : kind === 'env' ? 'hsl(150, 55%, 42%)' : 'var(--accent-2)'

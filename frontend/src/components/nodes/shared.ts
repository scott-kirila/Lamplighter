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

// The title ON a data node's header fill. Same measured pairing as the layer
// nodes (see lib/nodeColor's nodeTextColor): near-black clears AA on the amber
// and green fills in both themes, but the dataset violet flips — dark
// --accent-2 is light enough to want near-black (6.26), light --accent-2 is
// deep enough to want white (4.81) — so that one is a token the themes set.
export const dataKindTextColor = (kind: string): string =>
  kind === 'noise' || kind === 'env' ? 'var(--node-fg)' : 'var(--data-dataset-fg)'

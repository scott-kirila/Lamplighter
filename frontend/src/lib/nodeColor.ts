// A node's display color as a theme-aware CSS token (defined in index.css for
// both themes). Colors are per-category; Input and Output share the "io"
// category but differ, so Output is special-cased. Replaces the fixed hex the
// backend registry ships (which couldn't adapt to light/dark).
export function nodeColor(category: string | undefined, type: string | undefined): string {
  if (type === 'Output') return 'var(--node-output)'
  switch (category) {
    case 'io':
      return 'var(--node-io)'
    case 'layers':
      return 'var(--node-layers)'
    case 'activations':
      return 'var(--node-activations)'
    case 'ops':
      return 'var(--node-ops)'
    default:
      return 'var(--node-default)'
  }
}

// The title color to put ON a node's fill. Derived from the fill token rather
// than threaded through as extra node data — `var(--node-ops)` pairs with
// `var(--node-ops-fg)`, so one string transform keeps them from drifting apart.
//
// Why per-category at all: white failed on every dark fill (activations
// measured 1.86:1, ops 1.94:1) because these hues are picked to be bright, and
// in the light theme the set is genuinely mixed — the deep violet wants white,
// the mid-lightness orange wants near-black. index.css holds the measured
// values for both themes; this just picks the right token.
export function nodeTextColor(color: string): string {
  const match = /^var\((--node-[a-z]+)\)$/.exec(color)
  return match ? `var(${match[1]}-fg)` : 'var(--node-fg)'
}

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

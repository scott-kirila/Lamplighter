// Display formatting for inferred shapes. Inference runs with the Input's
// placeholder batch of 1, so a leading 1 on a multi-dim shape *is* the batch
// dim flowing through — render it as "N" so badges read "any batch × features"
// instead of looking like a literal size. `substitute=false` keeps raw numbers
// (used for pins that aren't batch-led, e.g. an LSTM's h_n = (layers, batch, h)).
export function formatShape(dims: number[], sep: string, substitute = true): string {
  const parts: (string | number)[] =
    substitute && dims.length > 1 && dims[0] === 1 ? ['N', ...dims.slice(1)] : dims
  return parts.join(sep)
}

// Beyond this many tensors the factorization stops explaining anything — a
// resnet18's 62 terms are a wall, not an insight. Hand-drawn layers are far
// under it (a Linear has 2, an LSTM 8).
const MAX_TERMS = 12

// A parameter count's factorization from the parameter tensors' shapes:
// Linear [[128, 784], [128]] → "128×784 + 128" (weight + bias). A rare scalar
// parameter (shape []) contributes "1". A deep pretrained backbone is
// summarized instead: the count above it is the number that matters.
export function formatParamTerms(terms: number[][]): string {
  if (terms.length > MAX_TERMS) return `${terms.length} tensors`
  return terms.map((t) => (t.length ? t.join('×') : '1')).join(' + ')
}

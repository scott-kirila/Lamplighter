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

// A parameter count's factorization from the parameter tensors' shapes:
// Linear [[128, 784], [128]] → "128×784 + 128" (weight + bias). A rare scalar
// parameter (shape []) contributes "1".
export function formatParamTerms(terms: number[][]): string {
  return terms.map((t) => (t.length ? t.join('×') : '1')).join(' + ')
}

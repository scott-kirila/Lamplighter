// Pure geometry for the training-health sparkline bars.

export interface SparkBar {
  i: number // the slot (epoch index within the span) this bar renders
  x: number
  y: number
  w: number
  h: number
}

/** Bar geometry for a fixed-width sparkline SVG. Every finite value gets a
 * bar — the x-axis spans `span` slots (the run's planned epochs), so bars
 * grow rightward as epochs stream and pixel density, not truncation, does
 * the compression, exactly like the loss charts' polylines. Non-finite
 * slots (e.g. the leading epochs a Δw/w series lacks) draw nothing. Bars
 * scale over the series' finite range; the minimum keeps a 1/8-height stub
 * so a flat series still reads as present. */
export function sparkBars(series: number[], span: number, width: number, height: number): SparkBar[] {
  const finite = series.filter(Number.isFinite)
  if (finite.length === 0) return []
  const n = Math.max(span, series.length, 1)
  const min = Math.min(...finite)
  const range = Math.max(...finite) - min || 1
  const w = width / n
  const bars: SparkBar[] = []
  series.forEach((v, i) => {
    if (!Number.isFinite(v)) return
    const h = (0.125 + 0.875 * ((v - min) / range)) * height
    bars.push({ i, x: i * w, y: height - h, w, h })
  })
  return bars
}

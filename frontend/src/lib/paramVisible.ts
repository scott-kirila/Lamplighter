import type { ParamDef } from '../types/graph'

// A param is shown unless its show_if names other params that don't all match the
// effective config. A rule value can be a single value (equality) or an array
// (membership — e.g. show for source in ["torchvision", "imagefolder"]).
// `effective` = stored values layered over the param defaults, so a gated field
// appears as soon as its controlling param matches — even before the user has
// touched that control. Shared by the Data and Training panels.
export function paramVisible(param: ParamDef, effective: Record<string, unknown>): boolean {
  if (!param.show_if) return true
  return Object.entries(param.show_if).every(([k, v]) =>
    Array.isArray(v) ? v.includes(effective[k]) : effective[k] === v
  )
}

// The Optimize picker's variant: a show_if-gated knob is OFFERABLE when its
// controller matches the effective config (the form rule above), OR when the
// controller is ITSELF being swept with a satisfying choice included —
// sweeping optimizer over [Adam, SGD] legitimately unlocks momentum (live in
// the SGD trials). Without this gate, momentum under a fixed Adam (or
// step_size under scheduler "none") becomes a silent no-op dimension:
// suggested, merged into the trial's training, and ignored by codegen —
// wasted trials and a meaningless crowned "best".
export function sweepOfferable(
  param: ParamDef,
  effective: Record<string, unknown>,
  swept: { name: string; choices?: string[] }[]
): boolean {
  if (!param.show_if) return true
  return Object.entries(param.show_if).every(([k, v]) => {
    const allowed = Array.isArray(v) ? v : [v]
    const controller = swept.find((s) => s.name === k)
    if (controller?.choices?.some((c) => allowed.includes(c))) return true
    return allowed.includes(effective[k])
  })
}

import type { ParamDef } from '../types/graph'

// A param is shown unless its show_if names other params that don't all match the
// effective config. `effective` = stored values layered over the param defaults,
// so a gated field appears as soon as its controlling param matches — even before
// the user has touched that control. Shared by the Data and Training panels.
export function paramVisible(param: ParamDef, effective: Record<string, unknown>): boolean {
  if (!param.show_if) return true
  return Object.entries(param.show_if).every(([k, v]) => effective[k] === v)
}

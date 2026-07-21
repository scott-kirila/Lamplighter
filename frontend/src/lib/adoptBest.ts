import type { SweepParamSpec } from './sweepScript'

export interface AdoptTargets {
  setTrainingParam: (key: string, value: unknown) => void
  patchNodeParam: (modelId: string, nodeId: string, param: string, value: unknown) => void
  patchDataParam: (nodeId: string, param: string, value: unknown) => void
}

// Map a finished sweep's winning params onto the project draft — the "adopt
// best" move: loop knobs merge into the training form, node-targeted specs
// patch their canvas node. Adopting IS drafting the next run from the winner
// (the form stays the next run's draft; the sweep's own records are
// untouched). A dotted key whose spec is gone (the draft was cleared since
// the sweep ran) is SKIPPED rather than polluting training with a bogus
// "<nodeId>.<param>" entry. Returns how many params were applied.
export function adoptBestParams(
  best: Record<string, unknown>,
  specs: SweepParamSpec[],
  targets: AdoptTargets
): number {
  let applied = 0
  for (const [key, value] of Object.entries(best)) {
    const spec = specs.find((s) => s.name === key)
    if (spec?.node) {
      targets.patchNodeParam(spec.node.model, spec.node.node, spec.node.param, value)
      applied += 1
    } else if (spec?.data) {
      targets.patchDataParam(spec.data.node, spec.data.param, value)
      applied += 1
    } else if (spec || !key.includes('.')) {
      targets.setTrainingParam(key, value)
      applied += 1
    }
  }
  return applied
}

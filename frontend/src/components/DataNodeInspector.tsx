import { useDataParams } from '../hooks/useDataParams'
import { paramVisible } from '../lib/paramVisible'
import { useGraphStore, type DataNodeMeta } from '../store/graphStore'
import type { ParamDef } from '../types/graph'
import { OptionalControl, ParamControl } from './Inspector'

// A noise source's params are small and fixed (frontend-defined). A dataset's
// come from /api/data/params — the same DATA_PARAMS the Data tab renders.
const NOISE_PARAMS: ParamDef[] = [
  // The per-sample latent size, e.g. "100" (or "100, 1, 1" for a conv generator).
  // No batch dim — the batch is drawn per step at run time.
  { name: 'dims', label: 'Latent dims', type: 'string', default: '100' },
  { name: 'distribution', label: 'Distribution', type: 'enum', default: 'normal', choices: ['normal', 'uniform'] },
]

// The Inspector for a data node selected on the system canvas: rename it and
// configure its source (a dataset's Data-panel form, or a noise node's shape).
export function DataNodeInspector({ node }: { node: DataNodeMeta }) {
  const setConfig = useGraphStore((s) => s.setDataNodeConfigParam)
  const renameDataNode = useGraphStore((s) => s.renameDataNode)
  const { data: dataParams } = useDataParams()

  const params = node.kind === 'noise' ? NOISE_PARAMS : dataParams ?? []
  const defaults = Object.fromEntries(params.map((p) => [p.name, p.default]))
  const effective: Record<string, unknown> = { ...defaults, ...node.config }

  return (
    <div
      style={{
        width: 300,
        flexShrink: 0,
        borderLeft: '1px solid var(--border)',
        background: 'var(--panel)',
        padding: 20,
        overflowY: 'auto',
        fontFamily: 'monospace',
      }}
    >
      <div style={{ color: 'var(--text-6)', fontSize: 10, letterSpacing: 1, textTransform: 'uppercase', marginBottom: 8 }}>
        {node.kind} source
      </div>
      <input
        value={node.name}
        onChange={(e) => renameDataNode(node.id, e.target.value)}
        style={{
          width: '100%', background: 'var(--field)', color: 'var(--text)', border: '1px solid var(--border)',
          borderRadius: 5, padding: '6px 8px', fontFamily: 'monospace', fontSize: 14, fontWeight: 700, marginBottom: 16,
        }}
      />
      {params
        .filter((p) => paramVisible(p, effective))
        .map((param) => {
          const props = {
            param,
            value: node.config[param.name],
            nodeColor: 'var(--accent-2)',
            onChange: (v: unknown) => setConfig(node.id, param.name, v),
          }
          return (
            <div key={param.name} style={{ marginBottom: 14 }}>
              <label style={{ display: 'block', color: 'var(--text-5)', fontSize: 11, marginBottom: 4 }}>
                {param.label}
              </label>
              {param.optional ? <OptionalControl {...props} /> : <ParamControl {...props} />}
            </div>
          )
        })}
    </div>
  )
}

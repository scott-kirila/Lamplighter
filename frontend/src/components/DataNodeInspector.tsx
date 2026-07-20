import { useDataParams } from '../hooks/useDataParams'
import { useDataVariables, type DataVariable } from '../hooks/useDataVariables'
import { eyebrow } from '../styles/ui'
import { useRecipes } from '../hooks/useRecipes'
import { paramVisible } from '../lib/paramVisible'
import { useGraphStore, type DataNodeMeta } from '../store/graphStore'
import type { ParamDef } from '../types/graph'
import { OptionalControl, ParamControl } from './ParamControl'

// A noise source's params are small and fixed (frontend-defined). A dataset's
// come from /api/data/params — the same DATA_PARAMS the Data tab rendered.
const NOISE_PARAMS: ParamDef[] = [
  // The per-sample latent size, e.g. "100" (or "100, 1, 1" for a conv generator).
  // No batch dim — the batch is drawn per step at run time.
  { name: 'dims', label: 'Latent dims', type: 'string', default: '100' },
  { name: 'distribution', label: 'Distribution', type: 'enum', default: 'normal', choices: ['normal', 'uniform'] },
]

// The curated discrete classic-control environments (mirrors backend RL_ENVS,
// which validates the pick at run/diagnose time — the backend is the source of
// truth; this list is the picker's affordance). The env's observation shape
// auto-fills the policy's Input; its action count is the policy's output logits.
const ENV_PARAMS: ParamDef[] = [
  {
    name: 'env_id', label: 'Environment', type: 'enum', default: 'CartPole-v1',
    choices: ['CartPole-v1', 'Acrobot-v1'],
  },
]

const SELECT_STYLE = {
  background: 'var(--field)', border: '1px solid var(--border)', borderRadius: 4,
  padding: '6px 8px', color: 'var(--text)', fontSize: 13, width: '100%',
  fontFamily: 'monospace', cursor: 'pointer',
} as const

// One label for a detected variable, e.g. "X — tensor [20, 8]".
function varLabel(v: DataVariable): string {
  const shape = v.shape ? ` [${v.shape.join(', ')}]` : v.num_samples != null ? ` (${v.num_samples})` : ''
  return `${v.name} — ${v.kind}${shape}`
}

// The memory-source picker: choose from the session's registered data for X (and
// y), pushing the inferred shape into the wired model's Input node(s). Leaving
// the picks empty is fine — codegen then emits a generic make_dataloaders(X, y).
function VariablePicker({
  node,
  needsTargets,
  modelId,
  modelNodes,
}: {
  node: DataNodeMeta
  needsTargets: boolean
  // The model this dataset is wired into — whose Input(s) receive X.
  modelId: string | undefined
  modelNodes: ReturnType<typeof useGraphStore.getState>['nodes']
}) {
  const setConfig = useGraphStore((s) => s.setDataNodeConfigParam)
  const updateNodeParamInModel = useGraphStore((s) => s.updateNodeParamInModel)
  const { data: variables, refetch, isFetching } = useDataVariables(true)
  const options = variables ?? []
  const config = node.config

  // Input nodes in forward-arg order (canvas position), matching model_inputs.
  const inputNodes = modelNodes
    .filter((n) => n.data.nodeType === 'Input')
    .sort((a, b) => a.position.y - b.position.y || a.position.x - b.position.x || a.id.localeCompare(b.id))
  const multi = inputNodes.length > 1

  // Applying a pick pushes the variable's inferred shape+dtype onto that Input
  // (of the wired model, active or stashed).
  const applyShape = (nodeId: string, varName: string) => {
    const v = options.find((o) => o.name === varName)
    if (!v?.input_shape || !modelId) return
    updateNodeParamInModel(modelId, nodeId, 'shape', v.input_shape.shape)
    updateNodeParamInModel(modelId, nodeId, 'dtype', v.input_shape.dtype)
  }

  // Per-input picks (multi-input), persisted in the node's config keyed by Input
  // node id — the runner resolves them (in forward-arg order) at run time.
  // Single-input keeps x_var (also read by codegen to detect a DataLoader pick).
  const picks = (config.x_vars ?? {}) as Record<string, string>
  const setPick = (nodeId: string, varName: string) =>
    setConfig(node.id, 'x_vars', { ...picks, [nodeId]: varName })

  const varSelect = (value: string, onPick: (v: string) => void, noneLabel = '— select —') => (
    <select value={value} onChange={(e) => onPick(e.target.value)} style={{ ...SELECT_STYLE, marginBottom: 10 }}>
      <option value="">{noneLabel}</option>
      {options.map((v) => (
        <option key={v.name} value={v.name}>{varLabel(v)}</option>
      ))}
    </select>
  )
  const label = (text: string) => (
    <label style={{ display: 'block', color: 'var(--text-5)', fontSize: 11, marginBottom: 4 }}>{text}</label>
  )
  const sectionHeader = (text: string) => (
    <div style={{ ...eyebrow, fontSize: 10, color: 'var(--text-8)', margin: '4px 0 8px' }}>
      {text}
    </div>
  )

  return (
    <div style={{ marginBottom: 16, paddingBottom: 12, borderBottom: '1px solid var(--border)' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
        <span style={{ color: 'var(--text-5)', fontSize: 11 }}>Registered data → Input shape</span>
        <button
          type="button"
          onClick={() => refetch()}
          style={{
            background: 'none', border: '1px solid var(--border)', borderRadius: 4,
            color: 'var(--text-4)', cursor: 'pointer', fontSize: 11, padding: '2px 8px', fontFamily: 'monospace',
          }}
        >
          {isFetching ? '…' : '↻ refresh'}
        </button>
      </div>

      {options.length === 0 ? (
        <div style={{ color: 'var(--text-7)', fontSize: 11, marginBottom: 8 }}>
          Nothing registered yet — run <span style={{ color: 'var(--accent)' }}>sess.data(X=X, y=y)</span> in
          the notebook, then refresh.
        </div>
      ) : null}

      {/* Input(s) — a named input still reads as an input under this header. */}
      {sectionHeader('Input(s)')}
      {multi ? (
        inputNodes.map((n, i) => {
          const name = typeof n.data.params.name === 'string' ? n.data.params.name.trim() : ''
          return (
            <div key={n.id}>
              {label(name || `Input ${i}`)}
              {varSelect(picks[n.id] ?? '', (v) => {
                setPick(n.id, v)
                applyShape(n.id, v)
              })}
            </div>
          )
        })
      ) : (
        varSelect(String(config.x_var ?? ''), (v) => {
          setConfig(node.id, 'x_var', v)
          if (inputNodes[0]) applyShape(inputNodes[0].id, v)
        })
      )}

      {/* An adversarial recipe (needs_targets=false) trains on X alone. */}
      {needsTargets && (
        <>
          {sectionHeader('Target(s)')}
          {varSelect(String(config.y_var ?? ''), (v) => setConfig(node.id, 'y_var', v), '— none / not needed —')}
        </>
      )}
    </div>
  )
}

// The Inspector for a data node selected on the overview canvas: rename it and
// configure its source (a dataset's Data-panel form, or a noise node's shape).
export function DataNodeInspector({ node }: { node: DataNodeMeta }) {
  const setConfig = useGraphStore((s) => s.setDataNodeConfigParam)
  const renameDataNode = useGraphStore((s) => s.renameDataNode)
  const { data: dataParams } = useDataParams()
  const { data: recipes } = useRecipes()

  // The model this data node is wired into (whose Input(s) get the picked shape),
  // and the recipe's data contract (a GAN's discriminator trains on X alone).
  const links = useGraphStore((s) => s.links)
  const nodes = useGraphStore((s) => s.nodes)
  const activeModelId = useGraphStore((s) => s.activeModelId)
  const modelGraphs = useGraphStore((s) => s.modelGraphs)
  const training = useGraphStore((s) => s.training)
  const modelId = links.find((l) => l.source_data === node.id)?.target_model
  const modelNodes = modelId ? (modelId === activeModelId ? nodes : modelGraphs[modelId]?.nodes ?? []) : []
  const recipe = recipes?.find((r) => r.name === ((training.recipe as string) ?? 'supervised'))
  const needsTargets = recipe?.needs_targets ?? true
  const hasVal = recipe?.has_val ?? true

  const params =
    node.kind === 'noise' ? NOISE_PARAMS : node.kind === 'env' ? ENV_PARAMS : dataParams ?? []
  const defaults = Object.fromEntries(params.map((p) => [p.name, p.default]))
  const effective: Record<string, unknown> = { ...defaults, ...node.config }
  const source = String(effective.source ?? 'memory')
  const isMemory = node.kind === 'dataset' && source === 'memory'

  // The memory source renders the registered-variable picker for x_var/y_var, so
  // drop those from the generic control list (no duplicate plain-text inputs);
  // hide the held-out split for a recipe that has none.
  const visibleParams = params.filter(
    (p) =>
      paramVisible(p, effective) &&
      !(isMemory && (p.name === 'x_var' || p.name === 'y_var')) &&
      !(p.name === 'val_split' && !hasVal)
  )

  const field = (param: ParamDef) => {
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
  }

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
      <div style={{ ...eyebrow, color: 'var(--text-6)', fontSize: 10, marginBottom: 8 }}>
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

      {/* Source selector first; the variable picker sits right under it. */}
      {visibleParams.filter((p) => p.name === 'source').map(field)}

      {isMemory && (
        <VariablePicker node={node} needsTargets={needsTargets} modelId={modelId} modelNodes={modelNodes} />
      )}

      {visibleParams.filter((p) => p.name !== 'source').map(field)}
    </div>
  )
}

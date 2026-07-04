import { beforeEach, describe, expect, it } from 'vitest'
import { epochsFromHistory, useGraphStore } from './graphStore'
import type { NodeDef } from '../types/graph'

// Minimal registry fixtures.
const INPUT: NodeDef = {
  type: 'Input', label: 'Input', category: 'io',
  inputs: [], outputs: [{ name: 'output', label: 'Out' }],
  params: [{ name: 'shape', label: 'Shape', type: 'shape', default: '1, 784' }],
}
const OUTPUT: NodeDef = {
  type: 'Output', label: 'Output', category: 'io',
  inputs: [{ name: 'input', label: 'In' }], outputs: [], params: [],
}
const RELU: NodeDef = {
  type: 'ReLU', label: 'ReLU', category: 'activations',
  inputs: [{ name: 'input', label: 'In' }], outputs: [{ name: 'output', label: 'Out' }], params: [],
}
const REGISTRY = { Input: INPUT, Output: OUTPUT, ReLU: RELU }

const store = useGraphStore.getState
const reset = () =>
  useGraphStore.setState({
    nodes: [],
    edges: [],
    selectedNodeId: null,
    training: {},
    data: {},
    activeTab: 'model',
    models: [{ id: 'model', name: 'Model', sysPosition: { x: 0, y: 0 } }],
    activeModelId: 'model',
    modelGraphs: {},
    modelResults: {},
    links: [],
  })

beforeEach(reset)

// Build an A -> B edge from two fresh nodes; returns their ids + the edge id.
function twoNodesConnected() {
  store().addNode(INPUT, { x: 0, y: 0 })
  store().addNode(OUTPUT, { x: 200, y: 0 })
  const [a, b] = store().nodes
  store().onConnect({ source: a.id, sourceHandle: 'output', target: b.id, targetHandle: 'input' })
  return { aId: a.id, bId: b.id, edgeId: store().edges[0].id }
}

describe('addNode', () => {
  it('seeds params from the node definition defaults', () => {
    store().addNode(INPUT, { x: 5, y: 6 })
    const n = store().nodes[0]
    expect(n.data.nodeType).toBe('Input')
    expect(n.data.params).toEqual({ shape: '1, 784' })
    expect(n.position).toEqual({ x: 5, y: 6 })
  })
})

describe('onConnect', () => {
  it('keeps a single edge per target input handle (replaces existing)', () => {
    store().addNode(RELU, { x: 0, y: 0 })
    store().addNode(RELU, { x: 0, y: 0 })
    store().addNode(OUTPUT, { x: 0, y: 0 })
    const [a, b, out] = store().nodes
    store().onConnect({ source: a.id, sourceHandle: 'output', target: out.id, targetHandle: 'input' })
    store().onConnect({ source: b.id, sourceHandle: 'output', target: out.id, targetHandle: 'input' })
    const into = store().edges.filter((e) => e.target === out.id && e.targetHandle === 'input')
    expect(into).toHaveLength(1)
    expect(into[0].source).toBe(b.id)
  })
})

describe('insertNodeOnEdge', () => {
  it('splices a new node into A->B, rewiring to A->N->B', () => {
    const { aId, bId, edgeId } = twoNodesConnected()
    store().insertNodeOnEdge(RELU, { x: 100, y: 0 }, edgeId)

    const nodes = store().nodes
    expect(nodes).toHaveLength(3)
    const newId = nodes.map((n) => n.id).find((id) => id !== aId && id !== bId)!

    const edges = store().edges
    expect(edges).toHaveLength(2)
    expect(edges.some((e) => e.id === edgeId)).toBe(false)
    expect(edges.some((e) => e.source === aId && e.target === newId && e.targetHandle === 'input')).toBe(true)
    expect(edges.some((e) => e.source === newId && e.sourceHandle === 'output' && e.target === bId)).toBe(true)
  })

  it('falls back to a plain add when the edge is gone', () => {
    store().insertNodeOnEdge(RELU, { x: 0, y: 0 }, 'does-not-exist')
    expect(store().nodes).toHaveLength(1)
    expect(store().edges).toHaveLength(0)
  })
})

describe('fit a spliced node (clear the source, make room before the target)', () => {
  // PITCH = NODE_WIDTH + INSERT_GAP = 230. A drop at x=100 on the same row as a
  // source at x=0 overlaps it, so the node is nudged to x=230; the target then
  // needs to sit at >= 230 + 230 = 460.
  const posOf = (id: string) => store().nodes.find((n) => n.id === id)!.position
  const insertedId = (before: string[]) =>
    store().nodes.map((n) => n.id).find((id) => !before.includes(id))!

  it('nudges the drop clear of the source and slides the right column over', () => {
    const { aId, bId, edgeId } = twoNodesConnected() // A at x=0, B at x=200
    store().addNode(RELU, { x: 200, y: 300 })        // parallel node in B's column
    const parallelId = store().nodes[2].id
    const before = store().nodes.map((n) => n.id)
    store().insertNodeOnEdge(RELU, { x: 100, y: 0 }, edgeId)

    expect(posOf(aId).x).toBe(0)                      // the source never moves
    expect(posOf(insertedId(before)).x).toBe(230)     // clear of the source
    expect(posOf(bId).x).toBe(460)                    // 230 + PITCH — minimum room
    expect(posOf(parallelId).x).toBe(460)             // same delta — layout preserved
  })

  it('keeps the dropped x when the drop is well below the source row', () => {
    const { bId, edgeId } = twoNodesConnected()
    const before = store().nodes.map((n) => n.id)
    store().insertNodeOnEdge(RELU, { x: 100, y: 300 }, edgeId) // no vertical overlap
    expect(posOf(insertedId(before)).x).toBe(100) // not clamped
    expect(posOf(bId).x).toBe(330)                // 100 + PITCH
  })

  it('moves nothing else when there is already room', () => {
    store().addNode(INPUT, { x: 0, y: 0 })
    store().addNode(OUTPUT, { x: 600, y: 0 })
    const [a, b] = store().nodes
    store().onConnect({ source: a.id, sourceHandle: 'output', target: b.id, targetHandle: 'input' })
    store().insertNodeOnEdge(RELU, { x: 300, y: 0 }, store().edges[0].id)
    expect(posOf(b.id).x).toBe(600)
  })

  it('leaves right-to-left layouts alone', () => {
    store().addNode(INPUT, { x: 400, y: 0 })
    store().addNode(OUTPUT, { x: 0, y: 0 })
    const [a, b] = store().nodes
    store().onConnect({ source: a.id, sourceHandle: 'output', target: b.id, targetHandle: 'input' })
    const before = store().nodes.map((n) => n.id)
    store().insertNodeOnEdge(RELU, { x: 200, y: 0 }, store().edges[0].id)
    expect(posOf(a.id).x).toBe(400)
    expect(posOf(b.id).x).toBe(0)
    expect(posOf(insertedId(before)).x).toBe(200) // dropped where the user put it
  })

  it('fits an existing node spliced in the same way', () => {
    const { bId, edgeId } = twoNodesConnected()
    store().addNode(RELU, { x: 100, y: 0 }) // dragged onto the wire, over the source
    const nId = store().nodes[2].id
    store().spliceNodeIntoEdge(nId, edgeId)
    expect(posOf(nId).x).toBe(230)  // nudged clear of the source
    expect(posOf(bId).x).toBe(460)  // downstream neighbor slid over
  })
})

describe('spliceNodeIntoEdge', () => {
  it('rewires an existing node into A->B', () => {
    const { aId, bId, edgeId } = twoNodesConnected()
    store().addNode(RELU, { x: 100, y: 0 })
    const nId = store().nodes[2].id

    store().spliceNodeIntoEdge(nId, edgeId)

    const edges = store().edges
    expect(edges).toHaveLength(2)
    expect(edges.some((e) => e.id === edgeId)).toBe(false)
    expect(edges.some((e) => e.source === aId && e.target === nId)).toBe(true)
    expect(edges.some((e) => e.source === nId && e.target === bId)).toBe(true)
  })
})

describe('seedDefault', () => {
  it('seeds an unconnected Input + Output', () => {
    store().seedDefault(REGISTRY)
    const types = store().nodes.map((n) => n.data.nodeType).sort()
    expect(types).toEqual(['Input', 'Output'])
    expect(store().edges).toHaveLength(0)
  })
})

describe('toDomainGraph / loadGraph round-trip', () => {
  it('reconstructs an equivalent domain graph', () => {
    twoNodesConnected()
    const before = store().toDomainGraph()

    store().loadGraph(before, REGISTRY)
    const after = store().toDomainGraph()

    expect(after).toEqual(before)
  })
})

describe('training config', () => {
  it('sets training params and includes them in the domain graph', () => {
    store().setTrainingParam('optimizer', 'SGD')
    store().setTrainingParam('lr', 0.05)
    expect(store().toDomainGraph().training).toEqual({ optimizer: 'SGD', lr: 0.05 })
  })

  it('round-trips training config through loadGraph', () => {
    store().setTrainingParam('epochs', 5)
    const d = store().toDomainGraph()
    store().loadGraph(d, REGISTRY)
    expect(store().training).toEqual({ epochs: 5 })
  })
})

describe('updateNodeParam', () => {
  it('updates a single param value', () => {
    store().addNode(INPUT, { x: 0, y: 0 })
    const id = store().nodes[0].id
    store().updateNodeParam(id, 'shape', '1, 28, 28')
    expect(store().nodes[0].data.params.shape).toBe('1, 28, 28')
  })
})

describe('models + toProject', () => {
  it('opens a model, switching to its canvas view', () => {
    store().setActiveTab('system')
    store().openModel('model')
    expect(store().activeModelId).toBe('model')
    expect(store().activeTab).toBe('model')
  })

  it('renames a model', () => {
    store().renameModel('model', 'Generator')
    expect(store().models[0].name).toBe('Generator')
  })

  it('tracks a model system-canvas position', () => {
    store().setModelSysPosition('model', { x: 40, y: 90 })
    expect(store().models[0].sysPosition).toEqual({ x: 40, y: 90 })
  })

  it('assembles a single-model project with the active graph and project config', () => {
    twoNodesConnected()
    store().setTrainingParam('lr', 0.05)
    store().setDataParam('source', 'memory')
    store().renameModel('model', 'Net')

    const project = store().toProject()
    expect(project.version).toBe(2)
    expect(project.links).toEqual([])
    expect(project.training).toEqual({ lr: 0.05 })
    expect(project.data).toEqual({ source: 'memory' })
    expect(project.models).toHaveLength(1)
    const [m] = project.models
    expect(m.id).toBe('model')
    expect(m.name).toBe('Net')
    // The active model carries the working graph; training/data live at the
    // project level, not on the model's graph.
    expect(m.graph.nodes).toHaveLength(2)
    expect(m.graph.edges).toHaveLength(1)
  })
})

describe('multiple models', () => {
  it('addModel seeds a new model, opens it, and stashes the old one', () => {
    twoNodesConnected() // the sole model now has 2 nodes
    const firstId = store().activeModelId

    store().addModel(REGISTRY)
    expect(store().models).toHaveLength(2)
    const secondId = store().activeModelId
    expect(secondId).not.toBe(firstId)
    expect(store().activeTab).toBe('model')
    // The new model is seeded Input + Output; the first model's graph is stashed.
    expect(store().nodes.map((n) => n.data.nodeType).sort()).toEqual(['Input', 'Output'])
    expect(store().modelGraphs[firstId].nodes).toHaveLength(2)
  })

  it('openModel swaps the active graph in and out (stashing is lossless)', () => {
    twoNodesConnected()
    const firstId = store().activeModelId
    store().addModel(REGISTRY)
    const secondId = store().activeModelId

    store().openModel(firstId)
    expect(store().activeModelId).toBe(firstId)
    expect(store().nodes).toHaveLength(2) // the first model's graph is back
    expect(store().modelGraphs[secondId].nodes).toHaveLength(2) // second stashed
  })

  it('deleteModel refuses the last model, and switches away when deleting the active', () => {
    store().deleteModel(store().activeModelId)
    expect(store().models).toHaveLength(1) // refused

    twoNodesConnected()
    const firstId = store().activeModelId
    store().addModel(REGISTRY)
    const secondId = store().activeModelId

    store().deleteModel(secondId) // delete the active (second) model
    expect(store().models.map((m) => m.id)).toEqual([firstId])
    expect(store().activeModelId).toBe(firstId)
    expect(store().nodes).toHaveLength(2) // reopened the first model's graph
  })

  it('toProject carries every model, and loadProject round-trips it', () => {
    twoNodesConnected()
    store().addModel(REGISTRY)
    store().renameModel(store().activeModelId, 'Discriminator')
    store().setTrainingParam('lr', 0.1)

    const project = store().toProject()
    expect(project.models).toHaveLength(2)
    expect(project.models.map((m) => m.graph.nodes.length).sort()).toEqual([2, 2])

    store().loadProject(project, REGISTRY)
    const after = store().toProject()
    expect(after.models).toHaveLength(2)
    expect(after.training).toEqual({ lr: 0.1 })
    expect(after.models.some((m) => m.name === 'Discriminator')).toBe(true)
  })

  it('setProjectResults routes the active model result into the flat maps', () => {
    const a = store().activeModelId
    store().setProjectResults(
      {
        [a]: { shapes: { n1: [4, 8] }, errors: {}, graph_issues: [] },
        other: { shapes: { z: [1] }, errors: { z: 'bad' }, graph_issues: ['x'] },
      },
      null
    )
    // The active model's shapes/errors are what the canvas reads.
    expect(store().shapes).toEqual({ n1: [4, 8] })
    // The other model's result is retained for when it becomes active.
    expect(store().modelResults.other.errors).toEqual({ z: 'bad' })
  })
})

describe('epochsFromHistory', () => {
  it('rebuilds the per-epoch stream from metric series', () => {
    const epochs = epochsFromHistory({ train_loss: [1, 0.5], val_loss: [0.9, 0.6] }, 10)
    expect(epochs).toEqual([
      { epoch: 1, epochs: 10, metrics: { train_loss: 1, val_loss: 0.9 } },
      { epoch: 2, epochs: 10, metrics: { train_loss: 0.5, val_loss: 0.6 } },
    ])
  })

  it('omits metrics whose series never ran (empty val without a val_loader)', () => {
    const epochs = epochsFromHistory({ train_loss: [1], val_loss: [] }, 5)
    expect(epochs).toEqual([{ epoch: 1, epochs: 5, metrics: { train_loss: 1 } }])
  })

  it('handles a null history (idle)', () => {
    expect(epochsFromHistory(null, 0)).toEqual([])
  })
})

describe('run hydration + event merging', () => {
  const e = (n: number) => ({ epoch: n, epochs: 5, metrics: { train_loss: 1 / n } })
  const resetRun = () =>
    useGraphStore.setState({ runState: 'idle', runEpochs: [], runError: null })

  it('appendRunEpoch ignores an epoch at/behind the newest (hydration race)', () => {
    resetRun()
    store().appendRunEpoch(e(1))
    store().appendRunEpoch(e(2))
    store().appendRunEpoch(e(2)) // duplicate delivery
    store().appendRunEpoch(e(1)) // stale
    expect(store().runEpochs.map((x) => x.epoch)).toEqual([1, 2])
  })

  it('hydrateRun seeds a late-joining tab', () => {
    resetRun()
    store().hydrateRun('running', null, [e(1), e(2), e(3)])
    expect(store().runState).toBe('running')
    expect(store().runEpochs).toHaveLength(3)
  })

  it('hydrateRun never downgrades live state or a longer epoch list', () => {
    resetRun()
    store().setRunStatus('done', null)
    store().appendRunEpoch(e(1))
    store().appendRunEpoch(e(2))
    // A stale fetch resolving late must not overwrite what the WS delivered.
    store().hydrateRun('running', null, [e(1)])
    expect(store().runState).toBe('done')
    expect(store().runEpochs).toHaveLength(2)
  })

  it('replaceRun overwrites the shown run wholesale (checkpoint restore)', () => {
    resetRun()
    store().setRunStatus('failed', 'boom', 99, null)
    store().appendRunEpoch(e(1))
    store().appendRunEpoch(e(2))
    // Restoring a checkpoint must replace everything — even a shorter history.
    store().replaceRun('done', null, [e(1)], 3, 1)
    expect(store().runState).toBe('done')
    expect(store().runError).toBeNull()
    expect(store().runEpochs).toHaveLength(1)
    expect(store().runSeed).toBe(3)
    expect(store().runBestEpoch).toBe(1)
  })
})

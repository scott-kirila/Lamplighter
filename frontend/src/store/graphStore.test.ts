import { beforeEach, describe, expect, it } from 'vitest'
import { useGraphStore } from './graphStore'
import { useRunStore } from './runStore'
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
    activeTab: 'model',
    models: [{ id: 'model', name: 'Model', sysPosition: { x: 0, y: 0 } }],
    activeModelId: 'model',
    modelGraphs: {},
    modelResults: {},
    links: [],
    linkResults: {},
    dataNodes: [],
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

describe('toDomainGraph', () => {
  it('serializes the canvas as the on-the-wire graph shape', () => {
    const { aId, bId } = twoNodesConnected()
    const d = store().toDomainGraph()
    expect(d.nodes.map((n) => n.id)).toEqual([aId, bId])
    expect(d.nodes[0].type).toBe('Input')
    expect(d.edges).toHaveLength(1)
    expect(d.edges[0]).toMatchObject({ source: aId, target: bId })
  })
})

describe('training config', () => {
  it('sets training params and includes them in the domain graph', () => {
    store().setTrainingParam('optimizer', 'SGD')
    store().setTrainingParam('lr', 0.05)
    expect(store().toDomainGraph().training).toEqual({ optimizer: 'SGD', lr: 0.05 })
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
    store().setActiveTab('overview')
    store().openModel('model')
    expect(store().activeModelId).toBe('model')
    expect(store().activeTab).toBe('model')
  })

  it('renames a model', () => {
    store().renameModel('model', 'Generator')
    expect(store().models[0].name).toBe('Generator')
  })

  it('tracks a model overview-canvas position', () => {
    store().setModelSysPosition('model', { x: 40, y: 90 })
    expect(store().models[0].sysPosition).toEqual({ x: 40, y: 90 })
  })

  it('assembles a single-model project with the active graph and project config', () => {
    twoNodesConnected()
    store().setTrainingParam('lr', 0.05)
    store().renameModel('model', 'Net')

    const project = store().toProject()
    expect(project.version).toBe(3)
    expect(project.data_nodes).toEqual([])
    expect(project.links).toEqual([])
    expect(project.training).toEqual({ lr: 0.05 })
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

  it('deleteModel drops links touching the gone model', () => {
    twoNodesConnected()
    const firstId = store().activeModelId
    store().addModel(REGISTRY)
    const secondId = store().activeModelId
    store().addLink(firstId, secondId) // first → second
    store().addDataNode('dataset')
    const dsId = store().dataNodes[0].id
    store().addLink(dsId, secondId) // data → second
    expect(store().links).toHaveLength(2)

    store().deleteModel(secondId)
    expect(store().links).toEqual([]) // both links touched the deleted model
    expect(store().dataNodes).toHaveLength(1) // the data node itself survives
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

  it('addLink connects two models, ignoring self-links and duplicates', () => {
    store().addLink('a', 'b')
    store().addLink('a', 'b') // duplicate — ignored
    store().addLink('c', 'c') // self-link — ignored
    expect(store().links).toHaveLength(1)
    expect(store().links[0]).toMatchObject({ source_model: 'a', target_model: 'b' })

    const id = store().links[0].id
    store().removeLink(id)
    expect(store().links).toHaveLength(0)
  })

  it('addLink defaults target_input to null (the sole input)', () => {
    store().addLink('a', 'b')
    expect(store().links[0].target_input).toBeNull()
  })

  it('addLink fans a data node out to distinct input ports, deduping per port', () => {
    store().addDataNode('dataset')
    const dId = store().dataNodes[0].id
    const mId = store().activeModelId
    store().addLink(dId, mId, 'noise') // → the noise port
    store().addLink(dId, mId, 'label') // same source, a different port → a second wire
    store().addLink(dId, mId, 'noise') // duplicate of the first port → ignored
    expect(store().links).toHaveLength(2)
    expect(store().links.map((l) => l.target_input).sort()).toEqual(['label', 'noise'])
  })

  const withOutputShape = (modelId: string, outNodeId: string, dims: number[]) =>
    useGraphStore.setState({
      modelResults: {
        [modelId]: { shapes: { [outNodeId]: dims }, pinShapes: {}, paramCounts: {}, errors: {}, graphIssues: [], code: null },
      },
    })

  it('addLink seeds the active target model Input shape from the source output', () => {
    store().addNode(INPUT, { x: 0, y: 0 })
    store().addNode(OUTPUT, { x: 200, y: 0 })
    const outNode = store().nodes.find((n) => n.data.nodeType === 'Output')!
    const sourceId = store().activeModelId
    withOutputShape(sourceId, outNode.id, [1, 500])

    store().addModel(REGISTRY) // the new model is active; its Input defaults to 1, 784
    const targetId = store().activeModelId
    expect(store().nodes.find((n) => n.data.nodeType === 'Input')!.data.params.shape).toBe('1, 784')

    store().addLink(sourceId, targetId)
    expect(store().nodes.find((n) => n.data.nodeType === 'Input')!.data.params.shape).toBe('1, 500')
  })

  it('addLink seeds a stashed target model Input from the source output', () => {
    store().addNode(INPUT, { x: 0, y: 0 })
    store().addNode(OUTPUT, { x: 200, y: 0 })
    const outNode = store().nodes.find((n) => n.data.nodeType === 'Output')!
    const sourceId = store().activeModelId
    store().addModel(REGISTRY)
    const targetId = store().activeModelId
    store().openModel(sourceId) // source active, target stashed
    withOutputShape(sourceId, outNode.id, [1, 500])

    store().addLink(sourceId, targetId)
    const seeded = store().modelGraphs[targetId].nodes.find((n) => n.data.nodeType === 'Input')!
    expect(seeded.data.params.shape).toBe('1, 500')
  })

  it('addLink seeds the named target port on a multi-input model, leaving others alone', () => {
    store().addNode(INPUT, { x: 0, y: 0 })
    store().addNode(OUTPUT, { x: 200, y: 0 })
    const outNode = store().nodes.find((n) => n.data.nodeType === 'Output')!
    const sourceId = store().activeModelId
    withOutputShape(sourceId, outNode.id, [1, 500])

    store().addModel(REGISTRY) // the new model is active with a default Input (1, 784)
    const targetId = store().activeModelId
    store().addNode(INPUT, { x: 0, y: 120 }) // a second input port
    const [portA, portB] = store().nodes.filter((n) => n.data.nodeType === 'Input')

    store().addLink(sourceId, targetId, portB.id) // wire onto portB specifically
    const inputs = store().nodes.filter((n) => n.data.nodeType === 'Input')
    expect(inputs.find((n) => n.id === portB.id)!.data.params.shape).toBe('1, 500')
    expect(inputs.find((n) => n.id === portA.id)!.data.params.shape).not.toBe('1, 500')
  })

  it('setLinkResults keys the backend shape-check by link id', () => {
    store().setLinkResults([
      { id: 'L1', ok: true, message: 'A → B: N × 784' },
      { id: 'L2', ok: false, message: 'mismatch' },
    ])
    expect(store().linkResults.L1.ok).toBe(true)
    expect(store().linkResults.L2.message).toBe('mismatch')
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

describe('data nodes', () => {
  it('addDataNode adds a dataset/noise with sensible defaults', () => {
    store().addDataNode('dataset')
    store().addDataNode('noise')
    const dn = store().dataNodes
    expect(dn.map((d) => d.kind)).toEqual(['dataset', 'noise'])
    expect(dn[0].name).toBe('Data')
    expect(dn[1].name).toBe('Noise')
    expect(dn[1].config.dims).toBe('100')
  })

  it('addLink wires a data node into a model (source_data), never into a data node', () => {
    store().addDataNode('dataset')
    const dId = store().dataNodes[0].id
    const mId = store().activeModelId
    store().addLink(dId, mId)
    expect(store().links).toHaveLength(1)
    expect(store().links[0]).toMatchObject({ source_data: dId, target_model: mId })
    store().addLink(mId, dId) // a model can't wire *into* a data node
    expect(store().links).toHaveLength(1)
  })

  it('removeDataNode drops it and any links from it', () => {
    store().addDataNode('dataset')
    const dId = store().dataNodes[0].id
    store().addLink(dId, store().activeModelId)
    store().removeDataNode(dId)
    expect(store().dataNodes).toHaveLength(0)
    expect(store().links).toHaveLength(0)
  })

  it('toProject carries data nodes and loadProject round-trips them', () => {
    store().addDataNode('noise')
    const project = store().toProject()
    expect(project.data_nodes).toHaveLength(1)
    expect(project.data_nodes[0].kind).toBe('noise')

    store().loadProject(project, REGISTRY)
    expect(store().dataNodes.map((d) => d.kind)).toEqual(['noise'])
  })

  it('ensureGanNoise provisions a noise node wired to the generator (dims from its Input)', () => {
    store().addNode(INPUT, { x: 0, y: 0 })
    store().updateNodeParam(store().nodes[0].id, 'shape', '1, 100')
    const genId = store().activeModelId
    store().ensureGanNoise(genId)

    const noise = store().dataNodes.find((d) => d.kind === 'noise')!
    expect(noise.config.dims).toBe('100')
    expect(store().links.some((l) => l.source_data === noise.id && l.target_model === genId)).toBe(true)

    store().ensureGanNoise(genId) // idempotent — no second noise node
    expect(store().dataNodes.filter((d) => d.kind === 'noise')).toHaveLength(1)
  })

  it('editing a noise node dims re-seeds the wired generator Input', () => {
    store().addNode(INPUT, { x: 0, y: 0 })
    store().updateNodeParam(store().nodes[0].id, 'shape', '1, 100')
    const genId = store().activeModelId
    store().ensureGanNoise(genId)
    const noiseId = store().dataNodes.find((d) => d.kind === 'noise')!.id

    store().setDataNodeConfigParam(noiseId, 'dims', '64')
    expect(store().nodes.find((n) => n.data.nodeType === 'Input')!.data.params.shape).toBe('1, 64')
  })

  it('ensureDatasetFor provisions a dataset node wired to the model (idempotent)', () => {
    const mId = store().activeModelId
    store().ensureDatasetFor(mId)

    const dataset = store().dataNodes.find((d) => d.kind === 'dataset')!
    expect(dataset.config.source).toBe('memory')
    expect(store().links.some((l) => l.source_data === dataset.id && l.target_model === mId)).toBe(true)

    store().ensureDatasetFor(mId) // idempotent — no second dataset node
    expect(store().dataNodes.filter((d) => d.kind === 'dataset')).toHaveLength(1)
  })

  it('ensureCganWiring provisions noise + a labeled dataset conditioning both models', () => {
    const mkInput = (id: string, name: string, y: number, shape: string) => ({
      id,
      type: 'model',
      position: { x: 0, y },
      data: { nodeType: 'Input', label: 'Input', color: '', inputPins: [], outputPins: [], params: { name, shape } },
    })
    // A conditional generator (noise + label) and discriminator (image + label);
    // the discriminator is stashed, to prove wiring reads inactive models too.
    useGraphStore.setState({
      models: [
        { id: 'g', name: 'Generator', sysPosition: { x: 0, y: 0 } },
        { id: 'd', name: 'Discriminator', sysPosition: { x: 0, y: 0 } },
      ],
      activeModelId: 'g',
      nodes: [mkInput('gn', 'noise', 0, '1, 100'), mkInput('gl', 'label', 100, '1')],
      modelGraphs: { d: { nodes: [mkInput('di', 'image', 0, '1, 8'), mkInput('dl', 'label', 100, '1')], edges: [] } },
      dataNodes: [],
      links: [],
    })

    store().ensureCganWiring('g', 'd')

    const noise = store().dataNodes.find((x) => x.kind === 'noise')!
    const dataset = store().dataNodes.find((x) => x.kind === 'dataset')!
    expect(noise.config.dims).toBe('100') // from the generator's noise Input
    // Four links: noise→gen.noise, X→disc.image, y→disc.label, y→gen.label.
    const links = store().links
    expect(links).toHaveLength(4)
    expect(links).toContainEqual(
      expect.objectContaining({ source_data: noise.id, target_model: 'g', target_input: 'gn' })
    )
    expect(links).toContainEqual(
      expect.objectContaining({ source_data: dataset.id, source_pin: 'x', target_model: 'd', target_input: 'di' })
    )
    expect(links).toContainEqual(
      expect.objectContaining({ source_data: dataset.id, source_pin: 'y', target_model: 'd', target_input: 'dl' })
    )
    expect(links).toContainEqual(
      expect.objectContaining({ source_data: dataset.id, source_pin: 'y', target_model: 'g', target_input: 'gl' })
    )

    store().ensureCganWiring('g', 'd') // idempotent — dataset already feeds the discriminator
    expect(store().links).toHaveLength(4)
    expect(store().dataNodes).toHaveLength(2)
  })
})

describe('resetProject', () => {
  it('returns the whole project to its first-open state', () => {
    // Build a messy multi-model project with wiring, data, and training config.
    twoNodesConnected()
    store().setTrainingParam('lr', 0.1)
    store().addModel(REGISTRY)
    store().addDataNode('noise')
    store().addLink(store().dataNodes[0].id, store().activeModelId)
    store().setSelectedDataNode(store().dataNodes[0].id)

    store().resetProject(REGISTRY)

    const s = store()
    expect(s.models).toHaveLength(1)
    expect(s.nodes.map((n) => n.data.nodeType)).toEqual(['Input', 'Output'])  // the scaffold
    expect(s.edges).toEqual([])  // scaffold nodes start unconnected
    expect(s.links).toEqual([])
    expect(s.dataNodes).toEqual([])
    expect(s.modelGraphs).toEqual({})
    expect(s.training).toEqual({})
    expect(s.selectedDataNodeId).toBeNull()
    expect(s.activeTab).toBe('model')
    // The reset is a structural change, so toProject() (what the validate push
    // sends) carries the clean slate — that's what overwrites the autosave.
    const p = s.toProject()
    expect(p.models).toHaveLength(1)
    expect(p.links).toEqual([])
    expect(p.training).toEqual({})
  })
})

describe('undo / redo', () => {
  const history = () => useGraphStore.setState({ past: [], future: [], _lastCaptureKey: null })
  const nodeCount = () => store().nodes.length

  it('undoes and redoes an added node', () => {
    history()
    store().addNode(INPUT, { x: 0, y: 0 })
    store().addNode(RELU, { x: 100, y: 0 })
    expect(nodeCount()).toBe(2)
    store().undo()
    expect(nodeCount()).toBe(1)
    store().redo()
    expect(nodeCount()).toBe(2)
    expect(store().nodes[1].data.nodeType).toBe('ReLU')
  })

  it('restores a deleted node (removals via onNodesChange capture)', () => {
    history()
    const { aId } = twoNodesConnected()
    useGraphStore.setState({ past: [], future: [] }) // only the delete in history
    store().onNodesChange([{ type: 'remove', id: aId }])
    expect(nodeCount()).toBe(1)
    store().undo()
    expect(nodeCount()).toBe(2)
    expect(store().nodes.some((n) => n.id === aId)).toBe(true)
  })

  it('does not record drag ticks or selection changes', () => {
    history()
    store().addNode(INPUT, { x: 0, y: 0 })
    const id = store().nodes[0].id
    useGraphStore.setState({ past: [], future: [] })
    store().onNodesChange([{ type: 'position', id, position: { x: 50, y: 50 } }])
    store().onNodesChange([{ type: 'select', id, selected: true }])
    expect(store().past).toHaveLength(0)
  })

  it('coalesces keystrokes in one param field into one undo step', () => {
    history()
    store().addNode(INPUT, { x: 0, y: 0 })
    const id = store().nodes[0].id
    useGraphStore.setState({ past: [], future: [], _lastCaptureKey: null })
    store().updateNodeParam(id, 'shape', '1, 7')
    store().updateNodeParam(id, 'shape', '1, 78')
    store().updateNodeParam(id, 'shape', '1, 784')
    expect(store().past).toHaveLength(1) // three keystrokes, one step
    store().undo()
    expect(store().nodes[0].data.params.shape).toBe('1, 784') // the pre-edit default, one step back
  })

  it('a new edit clears the redo branch', () => {
    history()
    store().addNode(INPUT, { x: 0, y: 0 })
    store().undo()
    expect(store().future).toHaveLength(1)
    store().addNode(RELU, { x: 0, y: 0 })
    expect(store().future).toHaveLength(0)
  })

  it('undoes a model deletion, wiring and all', () => {
    history()
    twoNodesConnected()
    store().addModel(REGISTRY)
    const secondId = store().activeModelId
    store().addDataNode('dataset')
    store().addLink(store().dataNodes[0].id, secondId)
    useGraphStore.setState({ past: [], future: [] })

    store().deleteModel(secondId) // drops the model AND its links
    expect(store().models).toHaveLength(1)
    expect(store().links).toHaveLength(0)
    store().undo()
    expect(store().models).toHaveLength(2)
    expect(store().links).toHaveLength(1)
    expect(store().activeModelId).toBe(secondId)
  })

  it('a new project is a history boundary — not undoable, clears the run dashboard', () => {
    history()
    twoNodesConnected()
    store().setTrainingParam('lr', 0.05)
    // A finished run's dashboard (in the run store) is showing.
    useRunStore.getState().replaceRun('done', null, [{ epoch: 1, epochs: 1, metrics: { train_loss: 0.5 } }], null, 1)

    store().resetProject(REGISTRY)

    // Fresh canvas AND a cleared dashboard — no ghost of the old project.
    expect(store().training).toEqual({})
    expect(useRunStore.getState().runState).toBe('idle')
    expect(useRunStore.getState().runEpochs).toEqual([])
    expect(useRunStore.getState().runBestEpoch).toBeNull()
    // File→New starts fresh: the old project isn't reachable via undo.
    expect(store().past).toHaveLength(0)
    store().undo()
    expect(store().training).toEqual({})
    expect(nodeCount()).toBe(2)
  })

  it('one link gesture (with shape seeding) is one undo step', () => {
    history()
    store().addNode(INPUT, { x: 0, y: 0 })
    store().addNode(OUTPUT, { x: 200, y: 0 })
    const outNode = store().nodes.find((n) => n.data.nodeType === 'Output')!
    const sourceId = store().activeModelId
    useGraphStore.setState({
      modelResults: {
        [sourceId]: { shapes: { [outNode.id]: [1, 500] }, pinShapes: {}, paramCounts: {}, errors: {}, graphIssues: [], code: null },
      },
    })
    store().addModel(REGISTRY)
    const targetId = store().activeModelId
    useGraphStore.setState({ past: [], future: [], _lastCaptureKey: null })

    store().addLink(sourceId, targetId) // link + seeds the target Input shape
    expect(store().nodes.find((n) => n.data.nodeType === 'Input')!.data.params.shape).toBe('1, 500')
    expect(store().past).toHaveLength(1) // internal seeding didn't double-count
    store().undo()
    expect(store().links).toHaveLength(0)
    expect(store().nodes.find((n) => n.data.nodeType === 'Input')!.data.params.shape).toBe('1, 784')
  })

  it('history is capped', () => {
    history()
    for (let i = 0; i < 60; i++) store().addNode(RELU, { x: i, y: 0 })
    expect(store().past.length).toBeLessThanOrEqual(50)
  })
})

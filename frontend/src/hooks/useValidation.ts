import { useCallback, useEffect, useMemo, useRef } from 'react'
import { useGraphStore } from '../store/graphStore'
import type { DomainGraph, NodeDef, NodeMove } from '../types/graph'

// Structural signature of a graph — identical format whether built from the
// local store or an incoming domain graph, so the two can be compared directly.
function keyFromDomain(graph: DomainGraph): string {
  const n = graph.nodes
    .map((dn) => `${dn.id}:${dn.type}:${JSON.stringify(dn.params)}`)
    .join('|')
  const e = graph.edges
    .map((de) => `${de.source}.${de.sourceHandle ?? 'output'}>${de.target}.${de.targetHandle ?? 'input'}`)
    .join('|')
  return `${n}__${e}`
}

export function useValidation(enabled: boolean, registry: Record<string, NodeDef> | undefined) {
  const wsRef = useRef<WebSocket | null>(null)
  const setValidationResult = useGraphStore((s) => s.setValidationResult)
  const toDomainGraph = useGraphStore((s) => s.toDomainGraph)
  const loadGraph = useGraphStore((s) => s.loadGraph)
  const setNodePositions = useGraphStore((s) => s.setNodePositions)
  const nodes = useGraphStore((s) => s.nodes)
  const edges = useGraphStore((s) => s.edges)

  const sendValidation = useCallback(() => {
    const ws = wsRef.current
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'validate', graph: toDomainGraph() }))
    }
  }, [toDomainGraph])

  // Sent on drag-end only — a lightweight move that skips shape inference.
  const sendMove = useCallback((moves: NodeMove[]) => {
    const ws = wsRef.current
    if (ws?.readyState === WebSocket.OPEN && moves.length > 0) {
      ws.send(JSON.stringify({ type: 'moves', nodes: moves }))
    }
  }, [])

  // Structural signature of the local canvas (positions excluded so dragging
  // doesn't re-validate).
  const structuralKey = useMemo(() => {
    const n = nodes
      .map((nd) => `${nd.id}:${nd.data.nodeType}:${JSON.stringify(nd.data.params)}`)
      .join('|')
    const e = edges
      .map((ed) => `${ed.source}.${ed.sourceHandle ?? 'output'}>${ed.target}.${ed.targetHandle ?? 'input'}`)
      .join('|')
    return `${n}__${e}`
  }, [nodes, edges])

  // Refs so the long-lived WebSocket handlers can see the latest values.
  const structuralKeyRef = useRef(structuralKey)
  structuralKeyRef.current = structuralKey
  const registryRef = useRef(registry)
  registryRef.current = registry
  // Key of the last graph applied from a remote tab — used to suppress the
  // outgoing validate that applying it would otherwise trigger (echo guard).
  const remoteKeyRef = useRef<string | null>(null)

  // Reconnecting WebSocket lifecycle — held until the canvas has hydrated
  // from the backend, so a freshly opened tab never overwrites the cached graph.
  useEffect(() => {
    if (!enabled) return
    let ws: WebSocket | null = null
    let reconnectTimer: number | undefined
    let unmounted = false

    const connect = () => {
      const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
      ws = new WebSocket(`${proto}://${window.location.host}/ws`)
      wsRef.current = ws

      ws.onopen = () => sendValidation()
      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data as string)
        if (msg.type === 'shapes') {
          setValidationResult(msg.shapes, msg.errors)
        } else if (msg.type === 'sync') {
          // Another tab changed the graph — mirror it here.
          const incoming = msg.graph as DomainGraph
          const incomingKey = keyFromDomain(incoming)
          if (incomingKey === structuralKeyRef.current) {
            // Same graph already on screen; just refresh shapes, don't rebuild.
            setValidationResult(msg.shapes, msg.errors)
            return
          }
          remoteKeyRef.current = incomingKey
          if (registryRef.current) loadGraph(incoming, registryRef.current)
          setValidationResult(msg.shapes, msg.errors)
        } else if (msg.type === 'moves') {
          // Another tab finished dragging — apply positions only (no re-validate).
          setNodePositions(msg.nodes as NodeMove[])
        } else if (msg.type === 'error') {
          console.error('[lamplighter] validation error:', msg.message)
        }
      }
      ws.onclose = () => {
        if (!unmounted) reconnectTimer = window.setTimeout(connect, 1000)
      }
    }
    connect()

    return () => {
      unmounted = true
      if (reconnectTimer) window.clearTimeout(reconnectTimer)
      ws?.close()
    }
  }, [enabled]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!enabled) return
    // Don't echo a graph we just applied from a remote tab.
    if (structuralKey === remoteKeyRef.current) return
    sendValidation()
  }, [structuralKey, sendValidation, enabled])

  return { sendMove }
}

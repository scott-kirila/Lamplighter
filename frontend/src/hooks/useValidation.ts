import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
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
  const setCode = useGraphStore((s) => s.setCode)
  const toDomainGraph = useGraphStore((s) => s.toDomainGraph)
  const loadGraph = useGraphStore((s) => s.loadGraph)
  const setNodePositions = useGraphStore((s) => s.setNodePositions)
  const nodes = useGraphStore((s) => s.nodes)
  const edges = useGraphStore((s) => s.edges)
  // Set once the backend says the session was stopped from the notebook. Unlike
  // a transient disconnect, this is terminal — we stop reconnecting and let the
  // UI show that the session is gone.
  const [sessionStopped, setSessionStopped] = useState(false)
  // True while the socket is down and retrying — surfaced after a couple of
  // failed attempts so a brief blip doesn't flash the indicator. Covers a
  // kernel restart/crash, where no explicit stop signal ever arrives.
  const [reconnecting, setReconnecting] = useState(false)
  // Bumped by reconnect() to re-run the socket effect with a fresh closure —
  // the only way to clear the terminal `stopped` flag after a notebook stop.
  const [reconnectNonce, setReconnectNonce] = useState(0)

  // Revive a tab parked on the "session stopped" overlay: drop the terminal
  // state and re-open the socket. If a new session is up on this port the tab
  // rejoins it (and re-seeds it with the design the browser still holds).
  const reconnect = useCallback(() => {
    setSessionStopped(false)
    setReconnectNonce((n) => n + 1)
  }, [])

  // Whether this tab has ever held content. A freshly opened (still empty) tab
  // must not push its blank canvas as the authoritative graph — that's what
  // wiped a populated tab during a reconnect. A real "delete all" still
  // propagates, since by then the tab has held content.
  const hadContentRef = useRef(false)
  if (nodes.length > 0) hadContentRef.current = true

  const sendValidation = useCallback(() => {
    const ws = wsRef.current
    const graph = toDomainGraph()
    if (graph.nodes.length === 0 && !hadContentRef.current) return
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'validate', graph }))
    }
  }, [toDomainGraph])

  // Sent on drag-end only — a lightweight move that skips shape inference.
  const sendMove = useCallback((moves: NodeMove[]) => {
    const ws = wsRef.current
    if (ws?.readyState === WebSocket.OPEN && moves.length > 0) {
      ws.send(JSON.stringify({ type: 'moves', nodes: moves }))
    }
  }, [])

  // Whether this tab's code panel is open. Held in a ref so the socket's onopen
  // can re-register the preference after a reconnect.
  const wantsCodeRef = useRef(false)

  // Tell the backend to start/stop generating code for this tab. On enable the
  // server pushes the current code straight back (no edit needed to populate).
  const setCodePreview = useCallback((enabled: boolean) => {
    wantsCodeRef.current = enabled
    const ws = wsRef.current
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'code_preview', enabled }))
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
    // Terminal close requested by the backend — suppresses the reconnect.
    let stopped = false
    // Consecutive failed connect attempts; the indicator shows past the threshold.
    let attempts = 0
    const RECONNECT_THRESHOLD = 2

    const connect = () => {
      const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
      ws = new WebSocket(`${proto}://${window.location.host}/ws`)
      wsRef.current = ws

      ws.onopen = () => {
        attempts = 0
        setReconnecting(false)
        // Re-register an open code panel so the backend resumes generating code
        // for this tab after a reconnect.
        if (wantsCodeRef.current) {
          ws?.send(JSON.stringify({ type: 'code_preview', enabled: true }))
        }
        sendValidation()
      }
      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data as string)
        if (msg.type === 'shapes') {
          setValidationResult(msg.shapes, msg.errors, msg.graph_issues ?? [], msg.code ?? null)
        } else if (msg.type === 'sync') {
          // Another tab changed the graph — mirror it here.
          const incoming = msg.graph as DomainGraph
          const incomingKey = keyFromDomain(incoming)
          if (incomingKey === structuralKeyRef.current) {
            // Same graph already on screen; just refresh shapes, don't rebuild.
            setValidationResult(msg.shapes, msg.errors, msg.graph_issues ?? [], msg.code ?? null)
            return
          }
          remoteKeyRef.current = incomingKey
          if (registryRef.current) loadGraph(incoming, registryRef.current)
          setValidationResult(msg.shapes, msg.errors, msg.graph_issues ?? [], msg.code ?? null)
        } else if (msg.type === 'code') {
          // Pushed when this tab opens its panel — populate without an edit.
          setCode(msg.code ?? null)
        } else if (msg.type === 'moves') {
          // Another tab finished dragging — apply positions only (no re-validate).
          setNodePositions(msg.nodes as NodeMove[])
        } else if (msg.type === 'session_stopped') {
          // The notebook tore down the session — stop retrying and surface it.
          stopped = true
          setReconnecting(false)
          setSessionStopped(true)
        } else if (msg.type === 'error') {
          console.error('[lamplighter] validation error:', msg.message)
        }
      }
      ws.onclose = () => {
        if (unmounted || stopped) return
        attempts += 1
        if (attempts >= RECONNECT_THRESHOLD) setReconnecting(true)
        reconnectTimer = window.setTimeout(connect, 1000)
      }
    }
    connect()

    return () => {
      unmounted = true
      if (reconnectTimer) window.clearTimeout(reconnectTimer)
      ws?.close()
    }
  }, [enabled, reconnectNonce]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!enabled) return
    // Don't echo a graph we just applied from a remote tab.
    if (structuralKey === remoteKeyRef.current) return
    sendValidation()
  }, [structuralKey, sendValidation, enabled])

  return { sendMove, sessionStopped, reconnecting, reconnect, setCodePreview }
}

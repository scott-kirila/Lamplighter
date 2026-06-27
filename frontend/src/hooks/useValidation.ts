import { useCallback, useEffect, useMemo, useRef } from 'react'
import { useGraphStore } from '../store/graphStore'

export function useValidation() {
  const wsRef = useRef<WebSocket | null>(null)
  const setValidationResult = useGraphStore((s) => s.setValidationResult)
  const toDomainGraph = useGraphStore((s) => s.toDomainGraph)
  const nodes = useGraphStore((s) => s.nodes)
  const edges = useGraphStore((s) => s.edges)

  const sendValidation = useCallback(() => {
    const ws = wsRef.current
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'validate', graph: toDomainGraph() }))
    }
  }, [toDomainGraph])

  // Reconnecting WebSocket lifecycle
  useEffect(() => {
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
        } else if (msg.type === 'error') {
          console.error('[scorch] validation error:', msg.message)
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
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Only the graph's structure and params affect shapes — not node positions.
  // Keying the effect on a structural signature avoids re-validating on drag.
  const structuralKey = useMemo(() => {
    const n = nodes
      .map((nd) => `${nd.id}:${nd.data.nodeType}:${JSON.stringify(nd.data.params)}`)
      .join('|')
    const e = edges
      .map((ed) => `${ed.source}.${ed.sourceHandle}>${ed.target}.${ed.targetHandle}`)
      .join('|')
    return `${n}__${e}`
  }, [nodes, edges])

  useEffect(() => {
    sendValidation()
  }, [structuralKey, sendValidation])
}

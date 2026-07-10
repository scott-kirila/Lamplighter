import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { epochsFromHistory, useGraphStore } from '../store/graphStore'
import type { DomainProject, NodeDef, NodeMove } from '../types/graph'

// Structural signature of a whole project — models (nodes/params/edges), links,
// and shared config; positions excluded so dragging doesn't re-validate.
// Identical whether built from the local store or an incoming project, so the
// two can be compared directly (the echo guard).
function keyFromProject(project: DomainProject): string {
  const models = project.models
    .map((m) => {
      const n = m.graph.nodes.map((dn) => `${dn.id}:${dn.type}:${JSON.stringify(dn.params)}`).join('|')
      const e = m.graph.edges
        .map((de) => `${de.source}.${de.sourceHandle ?? 'output'}>${de.target}.${de.targetHandle ?? 'input'}`)
        .join('|')
      return `${m.id}~${n}~${e}`
    })
    .join('#')
  const links = project.links
    .map(
      (l) =>
        `${l.source_data ?? l.source_model ?? ''}.${l.source_pin ?? ''}>${l.target_model}.${l.target_input ?? ''}`
    )
    .join('|')
  const dataNodes = (project.data_nodes ?? [])
    .map((d) => `${d.id}:${d.kind}:${JSON.stringify(d.config)}`)
    .join('|')
  return `${models}__${dataNodes}__${links}__${JSON.stringify(project.training ?? {})}`
}

export function useValidation(enabled: boolean, registry: Record<string, NodeDef> | undefined) {
  const wsRef = useRef<WebSocket | null>(null)
  const setProjectResults = useGraphStore((s) => s.setProjectResults)
  const setProjectCode = useGraphStore((s) => s.setProjectCode)
  const setLinkResults = useGraphStore((s) => s.setLinkResults)
  const setRunStatus = useGraphStore((s) => s.setRunStatus)
  const appendRunEpoch = useGraphStore((s) => s.appendRunEpoch)
  const hydrateRun = useGraphStore((s) => s.hydrateRun)
  const queryClient = useQueryClient()
  const toProject = useGraphStore((s) => s.toProject)
  const loadProject = useGraphStore((s) => s.loadProject)
  const applyModelMoves = useGraphStore((s) => s.applyModelMoves)
  const applySystemMoves = useGraphStore((s) => s.applySystemMoves)
  const nodes = useGraphStore((s) => s.nodes)
  const edges = useGraphStore((s) => s.edges)
  const models = useGraphStore((s) => s.models)
  const modelGraphs = useGraphStore((s) => s.modelGraphs)
  const links = useGraphStore((s) => s.links)
  const dataNodes = useGraphStore((s) => s.dataNodes)
  const activeModelId = useGraphStore((s) => s.activeModelId)
  const training = useGraphStore((s) => s.training)
  // Set once the backend says the session was stopped from the notebook. Unlike
  // a transient disconnect, this is terminal — we stop reconnecting and let the
  // UI show that the session is gone.
  const [sessionStopped, setSessionStopped] = useState(false)
  // An unexpected backend exception while processing a WS message: the tab's
  // results are stale until the next successful validate. Shown as a banner.
  const [validationError, setValidationError] = useState<string | null>(null)
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
  // must not push its blank canvas as the authoritative project — that's what
  // wiped a populated tab during a reconnect. Multiple models, or any node in
  // the active model, counts as content.
  const hadContentRef = useRef(false)
  if (nodes.length > 0 || models.length > 1) hadContentRef.current = true

  const sendValidation = useCallback(() => {
    const ws = wsRef.current
    const project = toProject()
    const empty = project.models.every((m) => m.graph.nodes.length === 0)
    if (empty && !hadContentRef.current) return
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'validate', project }))
    }
  }, [toProject])

  // Sent on drag-end only — a lightweight move within the active model's canvas
  // that skips shape inference.
  const sendMove = useCallback(
    (moves: NodeMove[]) => {
      const ws = wsRef.current
      if (ws?.readyState === WebSocket.OPEN && moves.length > 0) {
        ws.send(JSON.stringify({ type: 'moves', model_id: activeModelId, nodes: moves }))
      }
    },
    [activeModelId]
  )

  // Drag-end on the system canvas — persists model sys_positions.
  const sendSystemMove = useCallback((moves: NodeMove[]) => {
    const ws = wsRef.current
    if (ws?.readyState === WebSocket.OPEN && moves.length > 0) {
      ws.send(JSON.stringify({ type: 'system_moves', nodes: moves }))
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

  // Structural signature of the local project (positions excluded so dragging
  // doesn't re-validate; switching the active model doesn't change it either).
  const structuralKey = useMemo(
    () => keyFromProject(toProject()),
    // Recompute whenever any model's structure, a data node, or the shared config changes.
    [nodes, edges, models, modelGraphs, dataNodes, links, training, activeModelId, toProject]
  )

  // Refs so the long-lived WebSocket handlers can see the latest values.
  const structuralKeyRef = useRef(structuralKey)
  structuralKeyRef.current = structuralKey
  const registryRef = useRef(registry)
  registryRef.current = registry
  // Key of the last project applied from a remote tab — used to suppress the
  // outgoing validate that applying it would otherwise trigger (echo guard).
  const remoteKeyRef = useRef<string | null>(null)

  // Reconnecting WebSocket lifecycle — held until the canvas has hydrated
  // from the backend, so a freshly opened tab never overwrites the cached project.
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
        // Late-join: seed any in-flight/finished run's state + history, since
        // this tab missed the earlier run_status/run_epoch broadcasts. The store
        // merges conservatively (live events win), so racing is safe.
        fetch('/api/run/status')
          .then((res) => (res.ok ? res.json() : null))
          .then((status) => {
            if (status && status.state !== 'idle') {
              hydrateRun(
                status.state,
                status.error ?? null,
                epochsFromHistory(status.history, status.epochs ?? 0),
                status.seed ?? null,
                status.best_epoch ?? null
              )
            }
          })
          .catch(() => {})
      }
      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data as string)
        if (msg.type === 'shapes') {
          setValidationError(null) // a successful validate — the backend recovered
          setProjectResults(msg.models ?? {}, msg.code ?? null)
          setLinkResults(msg.links ?? [])
        } else if (msg.type === 'sync') {
          // Another tab changed the project — mirror it here.
          const incoming = msg.project as DomainProject
          const incomingKey = keyFromProject(incoming)
          if (incomingKey === structuralKeyRef.current) {
            // Same structure already on screen; just refresh shapes, don't rebuild.
            setProjectResults(msg.models ?? {}, msg.code ?? null)
            setLinkResults(msg.links ?? [])
            return
          }
          remoteKeyRef.current = incomingKey
          if (registryRef.current) loadProject(incoming, registryRef.current)
          setProjectResults(msg.models ?? {}, msg.code ?? null)
          setLinkResults(msg.links ?? [])
        } else if (msg.type === 'code') {
          // Pushed when this tab opens its panel — populate without an edit.
          setProjectCode(msg.code ?? {})
        } else if (msg.type === 'moves') {
          // Another tab finished dragging within a model — apply positions only.
          applyModelMoves(msg.model_id ?? null, msg.nodes as NodeMove[])
        } else if (msg.type === 'system_moves') {
          applySystemMoves(msg.nodes as NodeMove[])
        } else if (msg.type === 'data_registry') {
          // sess.data(...) changed the registry — update the picker's list in
          // place, so registered data appears without hitting ↻ refresh.
          queryClient.setQueryData(['data-variables'], msg.variables)
        } else if (msg.type === 'checkpoints') {
          // The checkpoint store changed (saved/deleted, from the app or the
          // notebook) — update the Training tab's list in place.
          queryClient.setQueryData(['checkpoints'], msg.checkpoints)
        } else if (msg.type === 'run_status') {
          // In-kernel training run transition (running/done/stopped/failed).
          setRunStatus(msg.state, msg.error ?? null, msg.seed ?? null, msg.best_epoch ?? null)
        } else if (msg.type === 'run_epoch') {
          appendRunEpoch({ epoch: msg.epoch, epochs: msg.epochs, metrics: msg.metrics })
        } else if (msg.type === 'session_stopped') {
          // The notebook tore down the session — stop retrying and surface it.
          stopped = true
          setReconnecting(false)
          setSessionStopped(true)
        } else if (msg.type === 'error') {
          // The backend failed to process a message (an unexpected validation
          // exception) — the canvas is showing STALE results, so say so in the
          // UI, not just the devtools console. Cleared by the next good frame.
          console.error('[lamplighter] validation error:', msg.message)
          setValidationError(String(msg.message ?? 'validation failed'))
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
    // Don't echo a project we just applied from a remote tab.
    if (structuralKey === remoteKeyRef.current) return
    sendValidation()
  }, [structuralKey, sendValidation, enabled])

  return {
    sendMove,
    sendSystemMove,
    sessionStopped,
    reconnecting,
    reconnect,
    setCodePreview,
    validationError,
    dismissValidationError: () => setValidationError(null),
  }
}

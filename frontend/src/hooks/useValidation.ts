import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { useGraphStore } from '../store/graphStore'
import { epochsFromHistory, useRunStore } from '../store/runStore'
import { useSweepStore, type SweepStatus } from '../store/sweepStore'
import type { DomainProject, NodeDef, NodeMove } from '../types/graph'

// Structural signature of a whole project — models (nodes/params/edges), links,
// and shared config; positions excluded so dragging doesn't re-validate.
// Identical whether built from the local store or an incoming project, so the
// two can be compared directly (the echo guard).
export function keyFromProject(project: DomainProject): string {
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
  const setRunStatus = useRunStore((s) => s.setRunStatus)
  const appendRunEpoch = useRunStore((s) => s.appendRunEpoch)
  const appendRunStep = useRunStore((s) => s.appendRunStep)
  const hydrateRun = useRunStore((s) => s.hydrateRun)
  const setSweepStatus = useSweepStore((s) => s.setSweepStatus)
  const hydrateSweep = useSweepStore((s) => s.hydrateSweep)
  const queryClient = useQueryClient()
  const toProject = useGraphStore((s) => s.toProject)
  const loadProject = useGraphStore((s) => s.loadProject)
  const applyModelMoves = useGraphStore((s) => s.applyModelMoves)
  const applyOverviewMoves = useGraphStore((s) => s.applyOverviewMoves)
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
  // rejoins it (and re-seeds it with the project the browser still holds).
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

  // Drag-end on the overview canvas — persists model sys_positions.
  const sendOverviewMove = useCallback((moves: NodeMove[]) => {
    const ws = wsRef.current
    if (ws?.readyState === WebSocket.OPEN && moves.length > 0) {
      ws.send(JSON.stringify({ type: 'overview_moves', nodes: moves }))
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
    // toProject() reads the whole store, so ESLint can't see these are used —
    // but they're exactly the structural inputs the key must recompute on
    // (positions and activeModel excluded on purpose).
    // eslint-disable-next-line react-hooks/exhaustive-deps
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
                epochsFromHistory(status.history, status.epochs ?? 0, status.health_history),
                status.seed ?? null,
                status.best_epoch ?? null,
                status.steps ?? [],
                status.step_total ?? 0,
                status.config ?? null,
                status.run_name ?? null
              )
            }
          })
          .catch(() => {})
        // Same late-join treatment for a sweep: seed its state so a refreshed
        // tab shows the sweep in flight (or the last one's outcome). The store
        // only applies this when idle — live events win.
        fetch('/api/sweep/status')
          .then((res) => (res.ok ? res.json() : null))
          .then((status) => {
            if (status && status.state !== 'idle') hydrateSweep(status as SweepStatus)
          })
          .catch(() => {})
      }
      ws.onmessage = (event) => {
        // An unparseable frame used to throw out of the handler and take the
        // rest of the run's stream with it — the dashboard froze at the last
        // good epoch while the run went on reporting "done". The backend now
        // sanitizes non-finite metrics, so this is the belt to that's braces:
        // say what happened rather than going quiet.
        // Deliberately un-annotated: this is an evolving `let`, so every branch
        // below keeps the exact inference it had when the parse was inline.
        let msg
        try {
          msg = JSON.parse(event.data as string)
        } catch {
          setValidationError('the kernel sent a message this tab could not read — the display may be behind')
          return
        }
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
        } else if (msg.type === 'overview_moves') {
          applyOverviewMoves(msg.nodes as NodeMove[])
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
          setRunStatus(msg.state, msg.error ?? null, msg.seed ?? null, msg.best_epoch ?? null, msg.config ?? null, msg.run_name ?? null)
        } else if (msg.type === 'run_epoch') {
          appendRunEpoch({ epoch: msg.epoch, epochs: msg.epochs, metrics: msg.metrics, health: msg.health, secs: msg.secs })
        } else if (msg.type === 'run_step') {
          appendRunStep(msg.step, msg.metrics ?? {}, msg.total ?? 0, msg.epoch_x ?? null)
        } else if (msg.type === 'sweep_status') {
          // The Optimize engine's state transitions + per-trial progress.
          const { type: _type, ...status } = msg
          setSweepStatus(status as SweepStatus)
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
    // Debounced: each keystroke in a param field is a structural change, and an
    // undebounced send runs full-project inference + an autosave write + a
    // broadcast per keystroke. A short trailing window batches a burst into one
    // validate; sendValidation reads toProject() at FIRE time, so the final
    // state is what's sent. Cleanup cancels on the next change (or a remote
    // sync arriving mid-window, which re-runs this effect and re-checks the
    // echo guard).
    const t = window.setTimeout(sendValidation, 150)
    return () => window.clearTimeout(t)
  }, [structuralKey, sendValidation, enabled])

  return {
    sendMove,
    sendOverviewMove,
    sessionStopped,
    reconnecting,
    reconnect,
    setCodePreview,
    validationError,
    dismissValidationError: () => setValidationError(null),
  }
}

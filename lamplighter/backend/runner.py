"""Run training inside the kernel, triggered from the web app.

The runner executes exactly the code the app shows: it ``exec``s the same
generated sources (``generate_module``, ``generate_dataloader``,
``generate_training``) that the preview panes display — no parallel
implementation. Data comes from the session's registry (``sess.data(X=X, y=y)``
in the notebook, picked by name in the Data tab), resolved up front so a bad
pick fails fast with a clear message instead of mid-run.

One run at a time. Training happens on a daemon thread (like the server
itself); per-epoch progress and state transitions are pushed to open editor
tabs via the WebSocket manager's fire-and-forget broadcast. Artifacts (the
trained model and metric history) stay in kernel memory, exposed to the
notebook through ``Session.model`` / ``Session.history``.
"""
from __future__ import annotations

import random
import threading
import traceback
import time
from datetime import datetime
from typing import Any, Callable

from .checkpoints import CHECKPOINT_VERSION
from .codegen import (
    class_name_for,
    exec_generated,
    generate_dataloader,
    generate_eval,
    generate_sampling,
    generate_module,
    layer_nodes,
    model_inputs,
)
from . import datastore
from .datastore import registry
from .inference import build_incoming, graph_issues
from .introspect import variable_kind
from .recipes import get_recipe
from .registry import default_data
from .schema import Graph, Project, resolve_data_config, resolve_env_config


def _exec_source(source: str, wanted: str, filename: str) -> Any:
    """Run generated source (via the audited ``codegen.exec_generated``
    chokepoint) and return the named object — the same pattern the notebook
    client (build_model/build_trainer) uses."""
    return exec_generated(source, filename)[wanted]


def _exec_model(source: str, filename: str) -> Any:
    """Build the model class from generated module source, found by its type
    rather than a fixed name — so a per-model class name (``Generator``,
    ``Discriminator``) works the same as the classic ``GeneratedModel``. The
    *last* ``nn.Module`` subclass wins: spliced Custom-node classes are emitted
    above the model class, which codegen always writes last."""
    import torch.nn as nn

    found = None
    for value in exec_generated(source, filename).values():
        if isinstance(value, type) and issubclass(value, nn.Module) and value is not nn.Module:
            found = value
    if found is None:
        raise ValueError("generated source defined no model class")
    return found


def rebuild_models(checkpoint: dict[str, Any], tag: str = "rebuild") -> dict[str, Any]:
    """Rebuild each role's model from a checkpoint's own generated source + final
    weights (fresh instances, eval mode). Caller-owned — nothing in the kernel is
    touched, so it's safe for a read-only preview of a stored run. Raises
    ValueError when the checkpoint kept no weights."""
    state_dicts = checkpoint.get("state_dicts")
    if state_dicts is None:
        raise ValueError("this run kept no weights")
    sources = checkpoint["snapshot"]["sources"]["models"]
    models: dict[str, Any] = {}
    for role, sd in state_dicts.items():
        cls = _exec_model(sources[role], f"<lamplighter-{tag}-{role}>")
        m = cls()
        m.load_state_dict(sd)
        models[role] = m.eval()
    return models


def _model_by_id(project: Project, model_id: str | None):
    return next((m for m in project.models if m.id == model_id), None)


# Activations whose "not firing" state is a ~0 output — the only ones where a
# persistently-zero unit is meaningfully "dead". (tanh/sigmoid/softmax/leaky/ELU
# rest elsewhere, so ≈0 there doesn't mean dead; see _register_activation_hooks.)
_ZERO_FLOOR_ACTIVATIONS = frozenset({"ReLU", "ReLU6", "GELU", "SiLU"})

# Min seconds between streamed per-step loss points — bounds the socket event
# rate on fast batch loops (a point every ~100ms is plenty for a live curve).
_STEP_EMIT_INTERVAL = 0.1
# Cap on the retained step points (the frontend keeps the same bound).
_STEP_HISTORY_LIMIT = 4000


def _finite(value: Any) -> Any:
    """``None`` for a non-finite float, the value otherwise — applied to every
    number leaving the runner.

    A diverged run reports ``nan``/``inf``, and that is the single most common
    thing this tool must SHOW you. But it can't survive the trip: Starlette's
    ``send_json`` writes a bare ``NaN`` token, which is not JSON, so the browser's
    ``JSON.parse`` throws and the whole frame is lost — the dashboard freezes at
    the last good epoch while the run goes on to report "done". The REST fallback
    fails the other way (``JSONResponse`` refuses to serialize it and 500s).
    ``null`` is valid JSON in both, so the epoch still arrives, the table still
    shows the row, and the chart draws a gap instead of blanking.
    """
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite(v) for v in value]
    return value


def _finite_only(emit: Callable[[dict], None]) -> Callable[[dict], None]:
    """Wrap an emit callback so no non-finite float ever reaches the socket."""
    def guarded(message: dict) -> None:
        emit(_finite(message))

    return guarded


def run_config_from(snapshot: dict[str, Any] | None) -> dict[str, Any] | None:
    """A compact summary of the config a run actually used, from its snapshot —
    the dashboard labels results with it (the form edits the *next* run and can
    drift from what's shown). Module-level so the run store's view endpoint can
    label ANY stored run, not just the manager's current one."""
    if not snapshot:
        return None
    t = snapshot.get("training") or {}
    out: dict[str, Any] = {
        "recipe": t.get("recipe") or "supervised",
        "epochs": t.get("epochs"),
        "device": snapshot.get("device"),
    }
    per_role = t.get("per_role") or {}
    lrs = {r: c.get("lr") for r, c in per_role.items() if isinstance(c, dict) and c.get("lr") is not None}
    if lrs:
        out["lrs"] = lrs
    elif t.get("lr") is not None:
        out["lr"] = t.get("lr")
    return out




class RunManager:
    """State machine for the single in-kernel training run.

    Threading contract: the lifecycle transitions (``start``/``resume``/
    ``restore``/``checkpoint``) hold ``self._lock`` — only one may run, and none
    overlaps a state read. The training thread's per-epoch hook (``_on_epoch``)
    is deliberately lock-FREE: it reassigns ``epoch``/``history``/``best_*`` on
    the hot path, where taking the lock would block a ``status()`` poll through a
    CPU weight-clone and a disk autosave. Those reassignments are individually
    atomic under the GIL and ``_merged`` builds fresh objects (never mutates in
    place), so a concurrent reader gets a coherent snapshot that may trail by at
    most one epoch — never a torn one. ``history`` is written before ``epoch`` so
    a reader never sees an epoch count ahead of the curve it can show."""

    def __init__(self, record_runs: bool = False) -> None:
        # Whether terminal runs auto-record into the run store (the module
        # singleton does; bare test managers don't pollute the global store).
        self._record_runs = record_runs
        # The name this run reserved at start (run-N) — status carries it so
        # the list can show the live run before its record exists.
        self.run_name: str | None = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_requested = False
        self.state: str = "idle"  # idle | running | done | stopped | failed
        self.error: str | None = None
        # The full traceback for `error` — the frames the one-line summary
        # drops. Shipped in status() so the UI can offer it behind a details
        # toggle and the user can paste it into an issue.
        self.error_traceback: str | None = None
        self.epoch: int | None = None
        self.epochs: int | None = None
        self.seed: int | None = None
        # The trained module(s). ``models`` is role → module (the general case,
        # e.g. a GAN's generator/discriminator); ``model`` is the single-model
        # convenience (the sole module, or None for a multi-model run).
        self.model: Any = None
        self.models: dict[str, Any] = {}
        self._live_models: dict[str, Any] = {}
        self.history: dict[str, list[float]] | None = None
        # Best-val tracking: CPU-cloned weights from the epoch with the lowest
        # val_loss (None without validation — then "final" is the only model).
        self.best_epoch: int | None = None
        self.best_state_dict: dict[str, Any] | None = None
        self._best_val = float("inf")
        self._live_model: Any = None  # in-flight model, for epoch-boundary capture
        # Resume continuity: a warm-started run reports epochs offset past the
        # checkpoint's, and its history merges onto the stored one — one curve.
        self._epoch_offset = 0
        self._base_history: dict[str, list[float]] = {}
        self._autosave_every = 0
        # Early stopping: stop once val_loss hasn't improved for this many
        # epochs (0 = off). Runner-side like autosave — _on_epoch returns False
        # and the generated loop just breaks; inert without validation, since
        # best_epoch never sets without a val_loss to judge by.
        self._early_stop_patience = 0
        # Epochs since val last improved, counted WITHIN this segment. Deriving
        # it from (epoch - best_epoch) breaks across a resume: the restored
        # best_epoch belongs to the previous segment, so a run that stopped at
        # 5 with its best at 3 resumes at 21 and computes a stall of 18 —
        # triggering early stop after a single epoch and reporting "done".
        # Resuming is an explicit request to keep going, so each segment gets a
        # fresh patience budget.
        self._epochs_since_best = 0
        # Wall-clock of the previous epoch boundary (perf_counter), for per-epoch
        # timing. Set just before training starts; touched only on the train thread.
        self._last_epoch_ts = 0.0
        # Last time a per-step loss was emitted — throttles the step stream so a
        # fast batch loop doesn't flood the socket. Train-thread only.
        self._last_step_emit = 0.0
        # The emitted step points, kept so a tab that joins (or refreshes)
        # mid/post-run can rebuild the step chart — parallel to health_history.
        # Bounded like the frontend buffer: halves its density at the cap.
        self._step_history: list[dict[str, Any]] = []
        # Batches per epoch for the current run segment (0 = unknown/iterable):
        # with _epoch_offset it maps a step onto the epoch axis — a resumed
        # segment's step 1 belongs at epoch offset + 1/spe, not at zero.
        self._steps_per_epoch = 0
        # Total batches this run will train — (epochs this run) × batches/epoch —
        # so the step chart can fix its x-axis up front. 0 = unknown (fall back to
        # auto-scaling on the streamed range).
        self._total_steps = 0
        # Per-layer training-health readout: layer_N -> canvas-node label (per
        # role), the previous epoch's per-layer weights (for the update ratio),
        # and the streamed per-epoch snapshots. Reassigned lock-free in _on_epoch
        # (same contract as history), so status() reads a consistent list.
        self._layer_map: dict[str, dict[str, Any]] = {}  # role -> layer_N -> LayerNode
        self._prev_weights: dict[str, dict[str, Any]] | None = None
        self._health_history: list[dict[str, Any]] = []
        # Dead-unit tracking for activation layers (no params, so the norm walk
        # skips them): forward hooks OR a per-unit "activated this epoch" mask;
        # a unit dead all epoch → a dead ReLU. All on the training thread, so no
        # lock. Reset each epoch; hooks torn down at run end.
        self._alive_masks: dict[str, dict[str, Any]] = {}  # role -> layer_N -> bool tensor
        self._hook_handles: list[Any] = []
        # Full reproducibility record of the current/last run: seed, resolved
        # device, effective configs, the graph, and the exact generated sources.
        self.snapshot: dict[str, Any] | None = None
        self._emit: Callable[[dict], None] = lambda message: None

    # -- public API ----------------------------------------------------------

    def start(
        self,
        project: Project,
        namespace: dict[str, Any] | None = None,
        emit: Callable[[dict], None] | None = None,
        source: str = "app",
        study: str | None = None,
    ) -> str | None:
        """Validate and launch a run for a project (one or more models — a GAN
        sends several). Returns an error message if the run can't start (already
        running, invalid graph, unassigned role, unresolvable data), else None.
        Data and codegen are resolved *now*, so the thread never touches the
        namespace and a bad pick fails before anything starts.

        ``source`` records where the run came from ("app" — the ▶ Run button;
        "sweep" — an Optimize trial; "notebook" — a notebook-driven bridge) and
        ``study`` groups a sweep's trials — both land in the snapshot, so the
        run store can filter/badge them."""
        ns = registry() if namespace is None else namespace
        if emit is None:
            from .ws import manager

            emit = manager.broadcast_threadsafe

        with self._lock:
            if self.state == "running":
                return "a run is already in progress — stop it first"

            recipe = get_recipe((project.training or {}).get("recipe"))
            if recipe is None:
                return f"unknown training recipe '{(project.training or {}).get('recipe')}'"
            assignment, err = self._assign_roles(project, recipe)
            if err is not None:
                return err
            # Each assigned model's graph must be codegen-ready.
            for role, mid in assignment.items():
                issues = graph_issues(_model_by_id(project, mid).graph)
                if issues:
                    prefix = f"{role}: " if len(assignment) > 1 else ""
                    return prefix + "; ".join(issues)

            cfg = {**{p.name: p.default for p in recipe.params}, **(project.training or {})}
            # Record the RESOLVED recipe name (an unset recipe means supervised)
            # — the snapshot self-describes, like the resolved seed below.
            cfg["recipe"] = recipe.name
            # Resolve the run's seed now so the snapshot is complete at start:
            # an unset seed is drawn at random AND recorded, so every run stays
            # reproducible. The thread applies it before anything touches RNG —
            # and an env recipe additionally bakes it into the generated loop
            # (episodes reset with it), so rollouts replay.
            raw_seed = cfg.get("seed")
            resolved_seed = random.randrange(2**31) if raw_seed is None else int(raw_seed)

            sole = len(project.models) <= 1
            try:
                model_sources = {
                    role: generate_module(m.graph, class_name=class_name_for(m.name, sole))
                    for role, mid in assignment.items()
                    if (m := _model_by_id(project, mid))
                }
            except ValueError as exc:  # a codegen refusal is a start error, not a crash
                return str(exc)

            # Imported models (sess.inspect) carry original weights kept in the
            # kernel. Resolve them here, eagerly inside the lock like everything
            # else, so the run thread never touches the datastore. A kernel
            # restart drops the weights — say so rather than silently training a
            # freshly-initialized copy.
            import_weights: dict[str, tuple[list, list[str]]] = {}
            for role, mid in assignment.items():
                m = _model_by_id(project, mid)
                if m is not None and m.imported is not None:
                    weights = datastore.import_weights(mid)
                    if weights is None:
                        return (
                            f"'{m.name}' was imported but its weights aren't in the "
                            f"kernel anymore (a restart clears them) — re-run "
                            f"sess.inspect(...) to bring them back."
                        )
                    import_weights[role] = (weights, m.imported.state_keys)

            if recipe.data == "env":
                # RL: the environment IS the data source, created inside the
                # generated train() — no loader path runs at all. Preflight the
                # optional dependency with the exact install hint.
                try:
                    import gymnasium  # noqa: F401
                except ImportError:
                    return (
                        "reinforcement learning needs Gymnasium — "
                        'pip install "lamplighter[rl]"'
                    )
                data_config = resolve_env_config(
                    project, assignment.get(recipe.data_role)
                ) or {}
                # Inject the resolved seed before generation so the source
                # shows the replayable value ("runs exactly the code it shows").
                gen_project = project.model_copy(deep=True)
                gen_project.training = {**(project.training or {}), "seed": resolved_seed}
                try:
                    call = {
                        "model_sources": model_sources,
                        "trainer_source": recipe.generate(gen_project),
                        "data_source": None,
                    }
                except ValueError as exc:
                    return str(exc)
                # The step stream carries per-EPISODE returns; size its axis.
                episodes = int(cfg.get("episodes_per_iter") or 0)
                call["steps_per_epoch"] = episodes
                call["total_steps"] = max(0, int(cfg["epochs"])) * episodes
                data_snapshot = dict(data_config)
            else:
                # The data feeding the recipe's data-fed model (a GAN's
                # discriminator, the model for supervised): the dataset node
                # wired into it. needs_targets comes from the recipe.
                data_model_id = assignment.get(recipe.data_role) or (
                    project.models[0].id if project.models else None
                )
                data_model = _model_by_id(project, data_model_id)
                data_config = resolve_data_config(project, data_model_id)
                # The recipe's shape contract. Diagnose blocks this in the app,
                # but a notebook run never sees diagnose — and the failure it
                # prevents is silent (a next-token loop fed tabular X/y trains
                # to completion and reports a meaningless perplexity).
                from .diagnose import source_mismatch

                mismatch = source_mismatch(
                    recipe, str({**default_data(), **data_config}["source"])
                )
                if mismatch is not None:
                    return f"{mismatch[0]} — {mismatch[1]}"
                data_graph = self._loader_graph(data_model.graph, project.links, data_model_id)
                try:
                    call = self._resolve_call(
                        data_graph, data_config, ns, needs_targets=recipe.needs_targets
                    )
                    # All codegen happens here, against the same namespace
                    # snapshot the data was resolved from — the thread only
                    # execs sources.
                    call["model_sources"] = model_sources
                    call["trainer_source"] = recipe.generate(project)
                    call["data_source"] = generate_dataloader(
                        data_graph, data_config, namespace=ns,
                        needs_targets=recipe.needs_targets, has_val=recipe.has_val,
                    )
                except ValueError as exc:
                    return str(exc)
                data_snapshot = {**default_data(), **data_config}

            call["seed"] = resolved_seed
            call["import_weights"] = import_weights
            device = str(cfg.get("device", "auto"))
            if device == "auto":
                from .registry import available_devices

                av = available_devices()
                device = "cuda" if "cuda" in av else "mps" if "mps" in av else "cpu"

            self.state = "running"
            self.error = None
            self.error_traceback = None
            self._epochs_since_best = 0
            self.epoch = None
            self.epochs = int(cfg["epochs"])
            self.seed = call["seed"]
            self.model = None
            self.models = {}
            self.history = None
            self.best_epoch = None
            self.best_state_dict = None
            self._best_val = float("inf")
            self._epoch_offset = 0
            self._steps_per_epoch = 0
            self._step_history = []  # a fresh run starts a fresh step curve
            self._base_history = {}
            self._autosave_every = int(cfg.get("autosave_every") or 0)
            self._early_stop_patience = int(cfg.get("early_stop_patience") or 0)
            self._prev_weights = None
            self._health_history = []
            self._alive_masks = {}
            # layer_N → canvas-node label per role, computed once (reused each
            # epoch to label the health rows by node rather than an opaque index).
            self._layer_map = {
                role: {ln.layer: ln for ln in layer_nodes(_model_by_id(project, mid).graph)}
                for role, mid in assignment.items()
            }
            call["recipe"] = recipe.name
            self.snapshot = self._build_snapshot(
                project, assignment, cfg, device, call, data_snapshot, source=source, study=study
            )
            self._reserve_run_name()
            self._stop_requested = False
            self._emit = _finite_only(emit)
            # Emit "running" BEFORE the thread starts, so a fast run can't push
            # its final status first (which would leave tabs stuck on stale state).
            self._emit_status()
            self._thread = threading.Thread(
                target=self._run, args=(call,), daemon=True, name="lamplighter-run"
            )
            self._thread.start()
        return None

    def _assign_roles(self, project: Project, recipe) -> tuple[dict[str, str] | None, str | None]:
        """Map each recipe role to a model id. A single-role recipe with a
        single-model project auto-assigns; otherwise the roles come from
        ``training.roles``. Returns (assignment, error)."""
        explicit = (project.training or {}).get("roles") or {}
        assignment: dict[str, str] = {}
        for role in recipe.roles:
            mid = explicit.get(role.role)
            if mid is None and len(recipe.roles) == 1 and len(project.models) == 1:
                mid = project.models[0].id  # the obvious single-model case
            if mid is None:
                return None, f"assign a model to the '{role.role}' role in the Training tab"
            if _model_by_id(project, mid) is None:
                return None, f"the '{role.role}' role points to a model that doesn't exist"
            assignment[role.role] = mid
        # One model per role: the same model as, say, both generator AND
        # discriminator would train it against itself with both losses — the
        # form makes this unrepresentable (role picks swap), and this is the
        # backstop for raw API callers.
        holders: dict[str, str] = {}
        for role_name, mid in assignment.items():
            if mid in holders:
                return None, (
                    f"the '{holders[mid]}' and '{role_name}' roles point at the same "
                    "model — assign each role its own model"
                )
            holders[mid] = role_name
        return assignment, None

    def _build_snapshot(
        self, project: Project, assignment: dict[str, str], cfg: dict, device: str,
        call: dict, data_snapshot: dict, source: str = "app", study: str | None = None,
    ) -> dict[str, Any]:
        """The run's reproducibility record: the whole ``project`` plus per-role
        ``sources.models`` (a sole model is the ``"model"`` role). ``data`` is the
        RESOLVED data record — the dataset config merged over the form defaults
        (loader recipes) or the wired env node's config as-is (env recipes) — so
        a resume rebuilds the same source. ``source``/``study`` tag where the
        run came from (a sweep's trials carry their study name)."""
        snapshot = {
            "seed": call["seed"],
            "device": device,
            "training": cfg,
            "data": dict(data_snapshot),
            "project": project.model_dump(),
            "sources": {
                "models": call["model_sources"],
                "data": call["data_source"],
                "trainer": call["trainer_source"],
            },
            "source": source,
            "started": datetime.now().isoformat(timespec="seconds"),
        }
        if study is not None:
            snapshot["study"] = study
        return snapshot

    def resume(
        self,
        name: str,
        checkpoint: dict[str, Any],
        epochs: int | None = None,
        namespace: dict[str, Any] | None = None,
        emit: Callable[[dict], None] | None = None,
    ) -> str | None:
        """Warm-start from a stored checkpoint, continuing toward its planned
        epoch target. ``epochs`` means what it means everywhere else — the
        run's TOTAL length: by default the checkpoint's own plan (so an
        interrupted/autosaved run finishes exactly where it was headed), or a
        higher target to extend a finished run. Already at the target →
        refused with the fix spelled out.

        Everything comes from the checkpoint's OWN snapshot (graph, config,
        sources — the live canvas is irrelevant; the weights must match the
        stored architecture). The model is rebuilt from the stored source and
        loaded with the final weights, then trained further with a fresh
        optimizer and a newly drawn (recorded) seed. Epoch numbering continues
        where the checkpoint left off and the history merges into one
        continuous curve. Data is re-resolved from the registry by the stored
        picks, so re-registered names repoint as always. Returns an error
        message or None."""
        ns = registry() if namespace is None else namespace
        if emit is None:
            from .ws import manager

            emit = manager.broadcast_threadsafe

        if checkpoint.get("state_dicts") is None:
            return "this run kept no weights — resume needs them; keep weights on a finished run first"
        with self._lock:
            if self.state == "running":
                return "a run is already in progress — stop it first"

            snapshot = checkpoint["snapshot"]
            cfg = dict(snapshot["training"])
            offset = int(checkpoint.get("epoch") or 0)  # epochs already trained
            plan = int(cfg["epochs"])  # the checkpoint's own target
            target = plan if epochs is None else int(epochs)
            if target <= offset:
                if epochs is None:
                    return (
                        f"'{name}' already completed its {plan}-epoch plan — pass a "
                        f"higher target to train further, e.g. epochs={offset + plan}"
                    )
                return (
                    f"epochs={target} is not past the {offset} epochs already "
                    f"trained — pick a higher target"
                )
            remaining = target - offset
            cfg["epochs"] = target
            recipe = get_recipe(cfg.get("recipe"))
            if recipe is None:
                return f"unknown training recipe '{cfg.get('recipe')}'"

            # Rebuild the project from the snapshot. The model
            # source(s) travel verbatim (the weights match them); the trainer is
            # regenerated with the REMAINING count baked in (a stored trainer
            # bakes its own run's count, rarely what's left). Data codegen re-runs
            # against the current namespace so repointed names keep working.
            project = Project.model_validate(snapshot["project"])
            model_sources = snapshot["sources"]["models"]
            # A warm start is a NEW run: fresh drawn seed, recorded like any
            # other (the stored seed already had its run). Drawn BEFORE codegen
            # because an env recipe bakes it into the regenerated loop.
            new_seed = random.randrange(2**31)
            roles = (project.training or {}).get("roles") or {}
            if recipe.data == "env":
                try:
                    import gymnasium  # noqa: F401
                except ImportError:
                    return (
                        "reinforcement learning needs Gymnasium — "
                        'pip install "lamplighter[rl]"'
                    )
                gen_project = project.model_copy(deep=True)
                gen_project.training = {
                    **(project.training or {}),
                    # The whole plan plus where we pick up — not `remaining` —
                    # so an LR schedule spans the run it was configured for
                    # instead of restarting its shape for the second segment.
                    "epochs": target, "start_epoch": offset, "seed": new_seed,
                }
                try:
                    call = {
                        "model_sources": model_sources,
                        "recipe": recipe.name,
                        "trainer_source": recipe.generate(gen_project),
                        "data_source": None,
                    }
                except ValueError as exc:
                    return str(exc)
                episodes = int(cfg.get("episodes_per_iter") or 0)
                call["steps_per_epoch"] = episodes
                call["total_steps"] = remaining * episodes
            else:
                # The loader is built from the recipe's data-fed model (minus any
                # label port — see _loader_graph), exactly as `start` does, so a
                # conditional resume yields (X, y) too. The snapshot carries the
                # RESOLVED data config, so the same loader is rebuilt whatever
                # node/form fed it.
                data_model_id = roles.get(recipe.data_role) or project.models[0].id
                data_model = _model_by_id(project, data_model_id)
                data_config = dict(snapshot.get("data") or {})
                data_graph = self._loader_graph(data_model.graph, project.links, data_model_id)
                try:
                    call = self._resolve_call(data_graph, data_config, ns, needs_targets=recipe.needs_targets)
                    call["model_sources"] = model_sources
                    call["recipe"] = recipe.name
                    gen_project = project.model_copy(deep=True)
                    gen_project.training = {
                        **(project.training or {}), "epochs": target, "start_epoch": offset,
                    }
                    call["trainer_source"] = recipe.generate(gen_project)
                    call["data_source"] = generate_dataloader(
                        data_graph, data_config, namespace=ns,
                        # has_val was omitted here, so it defaulted to True and a
                        # recipe that never validates (a GAN) had a validation
                        # split carved out of its training data on resume that
                        # the original segment never had — the same run,
                        # silently trained on less.
                        needs_targets=recipe.needs_targets, has_val=recipe.has_val,
                    )
                except ValueError as exc:
                    return str(exc)

            call["state_dicts"] = checkpoint["state_dicts"]
            call["seed"] = new_seed
            cfg["seed"] = call["seed"]

            self.state = "running"
            self.error = None
            self.error_traceback = None
            self._epochs_since_best = 0
            self.epoch = offset
            self.epochs = target
            self.seed = call["seed"]
            self.model = None
            self.models = {}
            self._prev_weights = None
            # Carry the checkpoint's health curve across the seam so the health
            # panel continues instead of resetting; new epochs append to it.
            self._health_history = list(checkpoint.get("health_history") or [])
            # Same for the step curve: the old segment's points keep their baked
            # epoch_x, and the resumed segment's append after them — one
            # continuous loss curve across the seam.
            self._step_history = list(checkpoint.get("steps") or [])
            self._alive_masks = {}
            resume_assignment, _ = self._assign_roles(project, recipe)
            self._layer_map = {
                role: {ln.layer: ln for ln in layer_nodes(_model_by_id(project, mid).graph)}
                for role, mid in (resume_assignment or {}).items()
            }
            self._epoch_offset = offset
            self._base_history = {
                k: list(v) for k, v in (checkpoint.get("history") or {}).items()
            }
            self.history = dict(self._base_history) or None
            # Best-so-far carries across the seam (single-model, has_val only); a
            # multi-model run has no best marker.
            self.best_epoch = checkpoint.get("best_epoch")
            self.best_state_dict = checkpoint.get("best_state_dict")
            val = self._base_history.get("val_loss") or []
            self._best_val = min(val) if val else float("inf")
            self._autosave_every = int(cfg.get("autosave_every") or 0)
            self._early_stop_patience = int(cfg.get("early_stop_patience") or 0)
            sources = {
                "models": model_sources,
                "data": call["data_source"],
                "trainer": call["trainer_source"],
            }
            self.snapshot = {
                "seed": call["seed"],
                "device": snapshot["device"],  # resolved when the original run started
                "training": cfg,
                "data": snapshot["data"],
                "project": snapshot["project"],
                "sources": sources,
                "started": datetime.now().isoformat(timespec="seconds"),
                "resumed_from": name,
                "resumed_at_epoch": offset,
                "source": "app",
            }
            # A resumed run records as a NEW, longer run — fresh history name.
            self._reserve_run_name()
            self._stop_requested = False
            self._emit = _finite_only(emit)
            self._emit_status()
            self._thread = threading.Thread(
                target=self._run, args=(call,), daemon=True, name="lamplighter-run"
            )
            self._thread.start()
        return None

    def stop(self) -> None:
        """Request a cooperative stop — honored at the next epoch boundary."""
        self._stop_requested = True

    def _reserve_run_name(self) -> None:
        """Assign this run its history name (run-N) at start, so the list can
        show the live run before its record exists at run end."""
        if not self._record_runs:
            return
        from . import checkpoints

        self.run_name = checkpoints.next_run_name()

    def run_config(self) -> dict[str, Any] | None:
        return run_config_from(self.snapshot)

    def status(self) -> dict[str, Any]:
        # Lock-free by design (see the class docstring): reads the training
        # thread's fields, which may trail by one epoch but are never torn.
        # _finite for the same reason the emit path uses it — a diverged run's
        # nan would make FastAPI's JSONResponse refuse the whole payload, so the
        # late-joining tab that most needs to see the divergence gets a 500.
        return _finite({
            "state": self.state,
            "error": self.error,
            "error_traceback": self.error_traceback,
            "epoch": self.epoch,
            "epochs": self.epochs,
            "seed": self.seed,
            "best_epoch": self.best_epoch,
            "history": self.history,
            # Per-epoch per-layer health snapshots (parallel to history), for tabs
            # that join mid/post-run. Reassigned as a whole in _on_epoch, so this
            # reference is always a complete list.
            "health_history": self._health_history,
            # The streamed step points (bounded, whole-run span, each with its
            # baked epoch_x) — rebuilds the loss curve on a late join/refresh.
            "steps": self._step_history,
            "step_total": self._total_steps,
            "config": self.run_config(),
            "run_name": self.run_name,
        })

    # -- evaluation on the held-out test split ---------------------------------

    def evaluate(self, checkpoint: dict[str, Any] | None = None,
                 ns: dict[str, Any] | None = None) -> dict[str, Any]:
        """Score a run on data it never trained on — the number you'd quote.

        Runs the run's OWN recorded config: its graph, its loss, its split
        fractions, with a stored run's model rebuilt from its own generated
        source. The DATA is re-resolved from the live registry (there's nowhere
        else to get it), so a re-registered variable means a different test set
        — which is why the result carries the sample count it actually scored.
        Raises ValueError with a user-facing message when a run can't be
        evaluated (no weights, no test split, a recipe with no held-out notion).
        """
        if checkpoint is not None:
            models = rebuild_models(checkpoint, tag="evaluate")
            snapshot = checkpoint.get("snapshot") or {}
        else:
            with self._lock:
                models = dict(self.models)
                snapshot = self.snapshot
            if not models or not snapshot:
                raise ValueError("no trained model in the kernel — run training first")

        project = Project.model_validate(snapshot["project"])
        recipe = get_recipe((project.training or {}).get("recipe"))
        if recipe is None or not recipe.has_val:
            label = recipe.label if recipe else "this recipe"
            raise ValueError(f"{label} has no held-out evaluation — its loop trains without a val/test split")

        assignment, _ = self._assign_roles(project, recipe)
        role = recipe.data_role
        if role not in models:
            raise ValueError(f"the run has no '{role}' model to evaluate")
        model_def = _model_by_id(project, (assignment or {}).get(role))
        if model_def is None:
            raise ValueError("the trained model is no longer in the project")

        data = {**default_data(), **resolve_data_config(project, model_def.id)}
        source = str(data.get("source", "memory"))
        # A torchvision dataset ships its own official test split — that IS the
        # val loader here (train=False), so there's nothing to carve.
        official = source == "torchvision"
        if not official and float(data.get("test_split", 0.0) or 0.0) <= 0.0:
            raise ValueError(
                "this run has no test split — set one on the dataset node, then train a run to evaluate"
            )

        ns = registry() if ns is None else ns
        data_graph = self._loader_graph(model_def.graph, project.links, model_def.id)
        call = self._resolve_call(data_graph, data, ns, needs_targets=recipe.needs_targets)
        make = _exec_source(
            generate_dataloader(data_graph, data, namespace=ns,
                                needs_targets=recipe.needs_targets, has_val=recipe.has_val),
            "make_dataloaders", "<lamplighter-evaluate-data>",
        )
        loaders = make(*call["loader_args"])
        test_loader, split_label = self._pick_test_loader(loaders, source)
        if test_loader is None:
            raise ValueError("the test split holds no samples — raise it, or register more data")

        evaluate = _exec_source(
            generate_eval(model_def.graph, project.training or {}),
            "evaluate", "<lamplighter-evaluate>",
        )
        result = evaluate(models[role], test_loader)
        result["split"] = split_label
        result["evaluated_at"] = datetime.now().isoformat(timespec="seconds")
        return result

    # -- sampling from a language model ----------------------------------------

    def generate(self, prompt: str = "", max_new_tokens: int = 200, temperature: float = 1.0,
                 checkpoint: dict[str, Any] | None = None) -> dict[str, Any]:
        """Sample a continuation from a trained language model — the preview a
        loss curve can't give you.

        The tokenizer travels in the run's OWN recorded data source (the
        vocabulary is baked into it), so a stored run still decodes to text
        without the notebook that made it. A run trained on pre-tokenized ids
        has no vocabulary to decode with, and says so."""
        if checkpoint is not None:
            models = rebuild_models(checkpoint, tag="generate")
            snapshot = checkpoint.get("snapshot") or {}
        else:
            with self._lock:
                models = dict(self.models)
                snapshot = self.snapshot
            if not models or not snapshot:
                raise ValueError("no trained model in the kernel — run training first")

        project = Project.model_validate(snapshot["project"])
        recipe = get_recipe((project.training or {}).get("recipe"))
        if recipe is None or recipe.name != "causal_lm":
            raise ValueError("only a language-model run can generate text")
        model = models.get(recipe.data_role)
        if model is None:
            raise ValueError("the run has no model to sample from")

        # The run's own recorded loader source carries the vocabulary, and
        # exec_generated is the audited chokepoint for running it.
        ns = exec_generated(
            (snapshot.get("sources") or {}).get("data") or "", "<lamplighter-tokenizer>"
        )
        if "decode" not in ns:
            raise ValueError(
                "this run trained on pre-tokenized ids, so there's no vocabulary to read "
                "samples back with — register the raw text instead to generate"
            )

        block = int((snapshot.get("data") or {}).get("block_size", 128) or 128)
        device = str((snapshot.get("training") or {}).get("device", "cpu"))
        if device == "auto":
            device = "cpu"  # sampling is one token at a time; the CPU is fine
        sampler = _exec_source(generate_sampling(block, device), "generate", "<lamplighter-generate>")

        ids = ns["encode"](prompt)
        if ids.numel() == 0:
            # Nothing to continue from: start at a random point in the
            # vocabulary rather than refusing (an empty prompt is a fine ask).
            import torch

            ids = torch.randint(0, len(ns["VOCAB"]), (1,))
        completion = sampler(
            model, ids, max_new_tokens=max(1, min(int(max_new_tokens), 2000)),
            temperature=float(temperature),
        )
        text = ns["decode"](completion)
        return {
            "prompt": prompt,
            "text": text,
            "completion": text[len(ns["decode"](ids)):],
            "temperature": float(temperature),
            "vocab_size": len(ns["VOCAB"]),
        }

    @staticmethod
    def _pick_test_loader(loaders: Any, source: str) -> tuple[Any, str]:
        """(the loader holding data the run never trained on, what to call it).

        A torchvision dataset ships its own official test split, and that IS the
        val loader here (built with train=False) — so there's nothing to carve
        and nothing to re-split. Every other source uses the third loader a
        configured test_split produces. Pure, because which data a reported
        score came from is the one thing an evaluation must not get wrong."""
        if source == "torchvision":
            return loaders[1], "official test split"
        return (loaders[2] if len(loaders) > 2 else None), "held-out test split"

    def preview(self, role: str | None = None, n: int = 16, ns: dict[str, Any] | None = None) -> dict[str, Any]:
        """A sample of the LIVE model's input → output on real data. See
        _preview_with for the generic behaviour."""
        with self._lock:
            models = dict(self.models)
            snapshot = self.snapshot
        return self._preview_with(models, snapshot, role=role, n=n, ns=ns)

    def preview_checkpoint(self, checkpoint: dict[str, Any], role: str | None = None, n: int = 16,
                           ns: dict[str, Any] | None = None) -> dict[str, Any]:
        """Preview a STORED run's outputs — rebuild its models from the checkpoint
        and forward sample inputs, without touching the live kernel model (so you
        can flip between runs freely). Raises ValueError for a weightless run."""
        models = rebuild_models(checkpoint, tag="preview")
        return self._preview_with(models, checkpoint["snapshot"], role=role, n=n, ns=ns)

    def _preview_with(self, models: dict[str, Any], snapshot: dict[str, Any] | None,
                      role: str | None = None, n: int = 16, ns: dict[str, Any] | None = None) -> dict[str, Any]:
        """A sample of a model's input → output on real data — generic across
        model types: each forward input is resolved from the data node wired to
        it (dataset rows, or drawn noise), forwarded under no_grad, and returned
        as raw tensors (the frontend renders by shape — no task logic). Returns
        {"error": ...} for the gentle cases (no run, data gone, nothing wired) so
        the panel shows a note rather than failing."""
        import torch

        if not models or snapshot is None:
            return {"error": "no trained model yet — run training first"}
        ns = registry() if ns is None else ns
        n = max(1, min(int(n), 64))

        project = Project.model_validate(snapshot["project"])
        recipe = get_recipe((project.training or {}).get("recipe"))
        assignment, _ = self._assign_roles(project, recipe)
        role = role or (next(iter(models)) if len(models) == 1 else None)
        if role is None or role not in models:
            return {"error": f"pick a model to preview (roles: {', '.join(models)})"}
        model = models[role]
        model_def = _model_by_id(project, (assignment or {}).get(role))
        if model_def is None:
            return {"error": "the trained model is no longer in the project"}

        try:
            inputs = self._sample_inputs(project, model_def, n, ns)
            target = self._sample_target(project, model_def.id, n, ns) if (recipe is None or recipe.needs_targets) else None
        except ValueError as exc:
            return {"error": str(exc)}

        device = next((p.device for p in model.parameters()), torch.device("cpu"))
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                out = model(*[t.to(device) for t in inputs])
        except Exception as exc:  # a bad sample shape etc. — surface, don't crash
            return {"error": f"inference failed: {type(exc).__name__}: {exc}"}
        finally:
            model.train(was_training)

        outputs = list(out) if isinstance(out, (tuple, list)) else [out]
        return {
            "role": role,
            "n": int(inputs[0].shape[0]) if inputs else n,
            "inputs": [self._encode_tensor(t) for t in inputs],
            "outputs": [self._encode_tensor(t) for t in outputs if isinstance(t, torch.Tensor)],
            "target": self._encode_tensor(target) if target is not None else None,
        }

    def _sample_inputs(self, project: Project, model_def: Any, n: int, ns: dict[str, Any]) -> list[Any]:
        """One sampled tensor per forward() input, in arg order, each from the data
        node wired to that Input (dataset rows / drawn noise)."""
        graph = model_def.graph
        node_map = {nd.id: nd for nd in graph.nodes}
        input_ids = model_inputs(graph, build_incoming(graph), node_map)
        if not input_ids:
            raise ValueError("the model has no inputs to sample")
        sole = len(input_ids) == 1
        out = []
        for inp_id in input_ids:
            link = next(
                (
                    ln
                    for ln in project.links
                    if ln.target_model == model_def.id
                    and ln.source_data is not None
                    and (ln.target_input == inp_id or (sole and ln.target_input is None))
                ),
                None,
            )
            if link is None:
                raise ValueError("a model input isn't wired to a data source — wire one on the Overview canvas")
            dn = next((d for d in project.data_nodes if d.id == link.source_data), None)
            if dn is None:
                raise ValueError("the wired data node is missing")
            out.append(self._sample_from_node(dn, link.source_pin, inp_id, n, ns))
        return out

    def _sample_from_node(self, dn: Any, source_pin: str | None, inp_id: str, n: int, ns: dict[str, Any]) -> Any:
        import torch

        cfg = dn.config or {}
        if dn.kind == "noise":
            dims = [int(t) for t in str(cfg.get("dims", "100")).split(",") if t.strip()] or [100]
            draw = torch.rand if str(cfg.get("distribution", "normal")) == "uniform" else torch.randn
            return draw(n, *dims)
        source = str(cfg.get("source", "memory") or "memory")
        if dn.kind == "dataset" and source != "memory":
            # There is nothing pickable on a torchvision/imagefolder node — say
            # so, instead of instructing an impossible "pick a variable".
            what = cfg.get("dataset") if source == "torchvision" else cfg.get("root")
            raise ValueError(
                f"preview can't sample the {source} source ({what}) — it draws from "
                "in-memory tensors registered with sess.data(...)"
            )
        var = cfg.get("y_var") if source_pin == "y" else ((cfg.get("x_vars") or {}).get(inp_id) or cfg.get("x_var"))
        return self._sample_var(var, n, ns)

    def _sample_target(self, project: Project, model_id: str, n: int, ns: dict[str, Any]) -> Any:
        for link in project.links:
            if link.target_model == model_id and link.source_data is not None:
                dn = next((d for d in project.data_nodes if d.id == link.source_data and d.kind == "dataset"), None)
                if dn is not None and (dn.config or {}).get("y_var"):
                    try:
                        return self._sample_var((dn.config or {}).get("y_var"), n, ns)
                    except ValueError:
                        return None
        return None

    @staticmethod
    def _sample_var(var: Any, n: int, ns: dict[str, Any]) -> Any:
        import torch

        var = str(var or "").strip()
        if not var:
            raise ValueError("no variable picked for a model input — pick one in the data node's Inspector")
        if var not in ns:
            raise ValueError(f"'{var}' isn't registered — run sess.data({var}=...) in the notebook")
        t = ns[var]
        if not isinstance(t, torch.Tensor):
            raise ValueError(f"preview needs an in-memory tensor for '{var}'")
        return t[: min(n, len(t))]

    @staticmethod
    def _encode_tensor(t: Any, max_sample: int = 8192) -> dict[str, Any]:
        """A tensor as {shape, data} for the frontend. Per-sample element counts
        over max_sample are truncated (so an exotic huge output can't ship a
        megabyte); images/vectors are well under."""
        t = t.detach().to("cpu").float()
        shape = list(t.shape)
        per = int(t[0].numel()) if t.ndim > 1 else 1
        truncated = per > max_sample
        if truncated and t.ndim > 1:
            t = t.reshape(t.shape[0], -1)[:, :max_sample]
            shape = list(t.shape)
        return {"shape": shape, "data": t.reshape(-1).tolist(), "truncated": truncated}

    def rollout(self, max_steps: int = 500, episode: int = 0) -> dict[str, Any]:
        """One episode rolled out with the LIVE policy — RL's preview: frames,
        per-step action probabilities, and the reward tally. Read-only.
        ``episode`` indexes reproducible variants (see ``_rollout_with``)."""
        with self._lock:
            models = dict(self.models)
            snapshot = self.snapshot
        return self._rollout_with(models, snapshot, max_steps=max_steps, episode=episode)

    def rollout_checkpoint(
        self, checkpoint: dict[str, Any], max_steps: int = 500, episode: int = 0
    ) -> dict[str, Any]:
        """Roll out a STORED run's policy (rebuilt from its saved weights) —
        the preview_checkpoint pattern, so you can flip between trials and
        watch each one behave. The kernel's live model is untouched."""
        models = rebuild_models(checkpoint, tag="rollout")
        return self._rollout_with(models, checkpoint["snapshot"], max_steps=max_steps, episode=episode)

    def _rollout_with(
        self, models: dict[str, Any], snapshot: dict[str, Any] | None,
        max_steps: int = 500, episode: int = 0,
    ) -> dict[str, Any]:
        """The rollout itself: reset the run's OWN env with the run's OWN seed
        (fork_rng — the kernel's RNG is never perturbed), sample actions from
        the policy exactly as training does, and record a filmstrip. Frames are
        stride-downscaled at capture and subsampled to a fixed budget; probs
        come from a display-layer softmax over the policy's logits.

        ``episode`` picks a reproducible variant: episode k plays under
        ``seed + k``, so 0 is the run's canonical replay and every other index
        is a genuinely different — but individually replayable — episode."""
        import os

        if not models or snapshot is None:
            return {"error": "no trained policy yet — run training first"}
        env_id = (snapshot.get("data") or {}).get("env_id")
        if not env_id:
            return {"error": "this run wasn't an environment (RL) run"}
        try:
            import gymnasium as gym
        except ImportError:
            return {"error": 'rollouts need Gymnasium — pip install "lamplighter[rl]"'}
        import torch

        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless pygame render

        policy = models.get("policy") or next(iter(models.values()))
        was_training = policy.training
        policy.eval()
        device = next((p.device for p in policy.parameters()), torch.device("cpu"))
        frames: list[Any] = []
        probs: list[list[float]] = []
        actions: list[int] = []
        running_return: list[float] = []
        total = 0.0
        try:
            env = gym.make(str(env_id), render_mode="rgb_array")
            seed = snapshot.get("seed")
            if seed is not None:
                seed = int(seed) + max(0, int(episode))
            with torch.random.fork_rng(devices=[]):
                if seed is not None:
                    torch.manual_seed(int(seed))
                obs, _ = env.reset(seed=None if seed is None else int(seed))
                for _ in range(max(1, int(max_steps))):
                    frame = env.render()
                    frames.append(frame[::5, ::5].copy())  # downscale AT capture
                    x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        dist = torch.distributions.Categorical(logits=policy(x))
                    action = int(dist.sample())
                    probs.append([round(float(p), 4) for p in dist.probs.squeeze(0).tolist()])
                    actions.append(action)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    total += float(reward)
                    running_return.append(round(total, 2))
                    if terminated or truncated:
                        break
            env.close()
        except Exception as exc:
            return {"error": f"rollout failed: {type(exc).__name__}: {exc}"}
        finally:
            policy.train(was_training)

        keep = self._filmstrip_indices(len(frames))
        return {
            "env_id": env_id,
            "steps": len(frames),
            "total_return": total,
            "frames": [self._encode_frame(frames[i]) for i in keep],
            "probs": [probs[i] for i in keep],
            "actions": [actions[i] for i in keep],
            "returns": [running_return[i] for i in keep],
        }

    @staticmethod
    def _filmstrip_indices(n: int, limit: int = 48) -> list[int]:
        """Evenly subsample n frames to the filmstrip budget — first and last
        always kept, so the strip spans the whole episode."""
        if n <= limit:
            return list(range(n))
        step = (n - 1) / (limit - 1)
        return [round(i * step) for i in range(limit)]

    @staticmethod
    def _encode_frame(frame: Any) -> dict[str, Any]:
        """An RGB frame as {h, w, data} — a flat uint8 list the frontend paints
        straight into a canvas (no image codec on either side)."""
        h, w = int(frame.shape[0]), int(frame.shape[1])
        return {"h": h, "w": w, "data": frame.reshape(-1).tolist()}

    def join(self, timeout: float | None = None) -> bool:
        """Wait for the current run's thread (tests). True if it finished."""
        t = self._thread
        if t is None:
            return True
        t.join(timeout)
        return not t.is_alive()

    def run_record(self) -> dict[str, Any]:
        """The run as a WEIGHTLESS record (curves/health/steps/snapshot) — what
        auto-recording stores for every terminal run, failed ones included, so
        it tolerates a model-less run. Keeping weights upgrades the entry via
        checkpoint()."""
        return {
            "version": CHECKPOINT_VERSION,
            "state_dicts": None,
            "best_state_dict": None,
            "best_epoch": self.best_epoch,
            "epoch": max((len(v) for v in (self.history or {}).values()), default=0),
            "history": {k: list(v) for k, v in (self.history or {}).items()},
            "health_history": list(self._health_history),
            "steps": list(self._step_history),
            "step_total": self._total_steps,
            "snapshot": self.snapshot,
        }

    def checkpoint(self) -> dict[str, Any]:
        """The trained weights + the run snapshot, as one torch-saveable dict.
        Self-contained: the snapshot carries the generated model source(s), so the
        checkpoint can be rebuilt anywhere via lamplighter.load_checkpoint().
        Carries the best-val weights too, when validation ran (single-model,
        has_val — best_* are None otherwise). One shape for every run:
        ``state_dicts`` keyed by role, a sole model under ``"model"``."""
        if not self.models:
            raise ValueError("no trained model yet — run training first")
        return {
            "version": CHECKPOINT_VERSION,
            "state_dicts": {role: m.state_dict() for role, m in self.models.items()},
            "best_state_dict": self.best_state_dict,
            "best_epoch": self.best_epoch,
            "epoch": max((len(v) for v in (self.history or {}).values()), default=0),
            "history": self.history,
            "health_history": list(self._health_history),
            "steps": list(self._step_history),
            "step_total": self._total_steps,
            "snapshot": self.snapshot,
        }

    def restore(self, checkpoint: dict[str, Any], name: str | None = None) -> str | None:
        """Repopulate the run artifacts from a stored checkpoint: the model is
        rebuilt from the checkpoint's own generated source + final weights, so
        sess.model, the weights download, and resume all behave as if that run
        had just finished. ``name`` is the restored run's store name — it
        becomes the kernel's current run, so status/hydrate report it and the
        runs list marks it as shown (even across a refresh). Returns an error
        message if refused (mid-run), else None. load_state_dict copies the
        weights in, so the store's entry stays isolated from whatever happens
        to the live model afterwards."""
        if checkpoint.get("state_dicts") is None:
            return "this run kept no weights — view its curves instead, or resume a run that did"
        with self._lock:
            if self.state == "running":
                return "a run is in progress — stop it before restoring a checkpoint"
            snapshot = checkpoint["snapshot"]
            history = checkpoint.get("history") or {}

            self.models = rebuild_models(checkpoint, tag="restore")
            models = self.models
            # The single-model convenience handle, and the best-val marker (only a
            # single-model has_val run records one — the fields are None otherwise).
            self.model = next(iter(models.values())) if len(models) == 1 else None
            self.best_epoch = checkpoint.get("best_epoch")
            self.best_state_dict = checkpoint.get("best_state_dict")
            val = history.get("val_loss") or []
            self._best_val = min(val) if val else float("inf")

            self.state = "done"
            self.error = None
            self.error_traceback = None
            self._epochs_since_best = 0
            self.epochs = checkpoint.get("epoch")
            if self.epochs is None:
                self.epochs = max((len(v) for v in history.values()), default=0)
            self.epoch = self.epochs
            self.seed = snapshot.get("seed")
            self.history = {k: list(v) for k, v in history.items()} or None
            self.snapshot = snapshot
            # Restore the run's health curve too, so a restored run shows the same
            # per-layer health it had (older checkpoints without it → empty).
            self._health_history = list(checkpoint.get("health_history") or [])
            # And its step curve, same reasoning — the points carry their own
            # baked epoch_x, so no mapping state needs restoring with them.
            self._step_history = list(checkpoint.get("steps") or [])
            self._total_steps = int(checkpoint.get("step_total") or 0)
            # The kernel now holds this run — status/hydrate report its name so
            # the runs list marks it as shown, consistent with a fresh finish.
            self.run_name = name
            self._prev_weights = None
        return None

    def best_model(self) -> Any:
        """Rebuild the best-val-epoch model from the run's own generated source
        (None when validation didn't run). Fresh instance, eval mode, CPU."""
        if self.best_state_dict is None or self.snapshot is None:
            return None
        # A best marker implies a single-model has_val run — its sole source.
        (source,) = self.snapshot["sources"]["models"].values()
        model_cls = _exec_model(source, "<lamplighter-best-model>")
        model = model_cls()
        model.load_state_dict(self.best_state_dict)
        return model.eval()

    # -- data resolution (pre-flight) -----------------------------------------

    def _resolve_call(
        self, graph: Graph, data_config: dict, ns: dict[str, Any], needs_targets: bool = True
    ) -> dict[str, Any]:
        """Resolve the arguments the generated make_dataloaders() needs from the
        notebook namespace (all data flows through it — one path). ``data_config``
        is the wired dataset node's config (source, picked variables). Returns a
        call spec consumed by the thread body; raises ValueError with a user-facing
        message otherwise. ``needs_targets=False`` (an adversarial recipe)
        resolves the input X alone — no target."""
        data = {**default_data(), **(data_config or {})}
        source = str(data["source"])  # memory | sequence | torchvision | imagefolder

        if source == "sequence":
            # One stream: raw text (the loader carries a character tokenizer)
            # or a tensor of ids already tokenized in the notebook.
            name = str(data.get("corpus_var", "") or "").strip()
            if not name:
                raise ValueError("no text or token stream picked — pick one on the dataset node (Models tab)")
            if name not in ns:
                raise ValueError(f"'{name}' is not registered — run sess.data({name}=...) in the notebook")
            if isinstance(ns[name], str):
                return {"loader_args": (ns[name],)}
            return {"loader_args": (self._resolve_tensor(name, "tokens", ns),)}
        if source == "memory":
            x_var = str(data.get("x_var", "") or "").strip()
            kind = variable_kind(x_var, ns) if x_var else None
            if kind in ("dataloader", "dataset"):
                return {"loader_args": (ns[x_var],)}
            if not needs_targets:
                # Unlabeled (GAN): batches of the input alone.
                x = self._resolve_tensor(data.get("x_var"), "input (X)", ns)
                return {"loader_args": (x,)}
            xs = self._resolve_inputs(graph, data, ns)
            y = self._resolve_tensor(data.get("y_var"), "target (y)", ns)
            return {"loader_args": (*xs, y)}
        # torchvision / imagefolder need no data arguments (may download in-run).
        return {"loader_args": ()}

    @staticmethod
    def _loader_graph(base: Graph, links, data_model_id) -> Graph:
        """The graph the data loader is built from: the data-fed model's graph,
        minus any Input wired from the dataset's ``y`` (label) pin. Those inputs
        are conditioning fed by the loader's target column, not independent X — so
        a conditional model (a cGAN's discriminator) yields ``(X, y)`` rather than
        ``(X0, X1, y)``. Byte-identical to the model's graph when nothing is
        label-wired (every existing supervised/GAN run)."""
        label_ids = {
            link.target_input
            for link in links
            if link.source_data is not None
            and link.target_model == data_model_id
            and link.source_pin == "y"
            and link.target_input
        }
        if not label_ids:
            return Graph(nodes=base.nodes, edges=base.edges)
        nodes = [n for n in base.nodes if n.id not in label_ids]
        edges = [e for e in base.edges if e.source not in label_ids and e.target not in label_ids]
        return Graph(nodes=nodes, edges=edges)

    def _resolve_inputs(self, graph: Graph, data: dict, ns: dict[str, Any]) -> list[Any]:
        """The picked input variable(s), ordered to match forward() args. Single
        input uses data.x_var; multi-input uses data.x_vars keyed by Input node
        id, ordered by canvas position at run time (robust to reordering)."""
        node_map = {n.id: n for n in graph.nodes}
        incoming = build_incoming(graph)
        input_ids = model_inputs(graph, incoming, node_map)

        x_vars = data.get("x_vars") or {}
        if len(input_ids) <= 1:
            # Picks may live in x_vars even for a single loader input: the UI keys
            # them per-Input when the *model* shows several, and the loader graph
            # can be a reduction of it (a cGAN discriminator minus its label input).
            name = data.get("x_var") or (x_vars.get(input_ids[0]) if input_ids else None)
            return [self._resolve_tensor(name, "input (X)", ns)]

        xs: list[Any] = []
        for i, nid in enumerate(input_ids):
            name = str(x_vars.get(nid, "") or "").strip()
            label = node_map[nid].params.get("name") or f"Input {i}"
            if not name:
                raise ValueError(f"no variable picked for {label} — pick it on the dataset node (Models tab)")
            xs.append(self._resolve_tensor(name, str(label), ns))
        return xs

    @staticmethod
    def _resolve_tensor(name: Any, what: str, ns: dict[str, Any]) -> Any:
        """A named notebook variable that must be a torch.Tensor."""
        name = str(name or "").strip()
        if not name:
            raise ValueError(f"no variable picked for the {what} — pick it on the dataset node (Models tab)")
        if name not in ns:
            raise ValueError(f"'{name}' is not registered — run sess.data({name}=...) in the notebook")
        kind = variable_kind(name, ns)
        if kind == "ndarray":
            raise ValueError(f"'{name}' is a numpy array — convert it with torch.from_numpy({name})")
        if kind != "tensor":
            raise ValueError(f"'{name}' is a {kind or 'non-tensor object'} — expected a torch.Tensor here")
        return ns[name]

    # -- the run itself (background thread) -----------------------------------

    def _run(self, call: dict[str, Any]) -> None:
        try:
            import torch

            # Seed inside a forked RNG scope, before ANYTHING touches the RNG —
            # model init, random_split, and shuffling all draw from it. The fork
            # restores the kernel's global (CPU) RNG state afterwards, so a run
            # never perturbs the notebook's own randomness. (The state is still
            # shared while the run executes: torch RNG ops run in cells *during*
            # a run interleave with it — avoid those for bit-exact replays.)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(call["seed"])

                # One module per assigned role, each from its own generated source.
                models: dict[str, Any] = {}
                for role, source in call["model_sources"].items():
                    cls = _exec_model(source, f"<lamplighter-run-{role}>")
                    models[role] = cls()
                # Imported models (sess.inspect): seed the freshly-generated
                # module with the original weights, positionally and shape-
                # checked, so a first run continues from the imported state
                # rather than a random init. Skipped on resume, where the
                # checkpoint's own trained weights win below.
                if not call.get("state_dicts"):
                    from .importer import seed_from_weights

                    for role, (values, keys) in (call.get("import_weights") or {}).items():
                        seed_from_weights(models[role], values, keys)
                # Warm start (resume): load the stored weights, keyed by role.
                for role, sd in (call.get("state_dicts") or {}).items():
                    models[role].load_state_dict(sd)
                self._live_models = models
                self._register_activation_hooks(models)
                # Single-model convenience for best-val capture / autosave; None
                # for a multi-model run (no per-role best tracking yet).
                self._live_model = next(iter(models.values())) if len(models) == 1 else None
                train = _exec_source(call["trainer_source"], "train", "<lamplighter-run-trainer>")
                if call.get("data_source"):
                    make = _exec_source(call["data_source"], "make_dataloaders", "<lamplighter-run-data>")
                    # (train, val) — plus a test loader once a test split is
                    # configured. Indexed, not unpacked: the test set is for
                    # evaluate(), and training never touches it.
                    loaders = make(*call["loader_args"])
                    train_loader, val_loader = loaders[0], loaders[1]
                else:
                    # An env recipe: the environment lives inside train() —
                    # there is no loader. The step axis (per-episode returns)
                    # was sized at start from iterations × episodes.
                    train_loader = val_loader = None
                recipe = get_recipe(call["recipe"])
                self._last_epoch_ts = time.perf_counter()  # start the epoch-timing clock
                self._last_step_emit = 0.0  # so the first step always emits
                # Total steps this run will take, for the step chart's fixed x-axis.
                # The loop runs (planned - already-trained) epochs; a loader without
                # __len__ (IterableDataset) leaves it 0 → the chart auto-scales.
                if train_loader is None:
                    self._steps_per_epoch = int(call.get("steps_per_epoch") or 0)
                    self._total_steps = int(call.get("total_steps") or 0)
                else:
                    try:
                        epochs_this_run = max(0, (self.epochs or 0) - self._epoch_offset)
                        self._steps_per_epoch = len(train_loader)
                        self._total_steps = epochs_this_run * self._steps_per_epoch
                    except TypeError:
                        self._steps_per_epoch = 0
                        self._total_steps = 0
                history = recipe.bind(train, models, train_loader, val_loader, self._on_epoch, self._on_step)

            with self._lock:
                self.models = models
                self.model = next(iter(models.values())) if len(models) == 1 else None
                self.history = self._merged(history)
                self.state = "stopped" if self._stop_requested else "done"
        except Exception as exc:  # surface anything from user data/generated code
            with self._lock:
                self.state = "failed"
                self.error = f"{type(exc).__name__}: {exc}"
                # Keep the traceback. exec_generated registers every generated
                # source with linecache precisely so frames inside it resolve to
                # real lines — but this is where those exceptions land, and a
                # one-line summary threw that away. The failures that reach here
                # are the ones diagnose.py can't pre-empt (a spliced custom
                # module, a dataset __getitem__, a dtype/device mismatch, OOM),
                # and they're unreadable without the frames. The run thread is a
                # daemon, so nothing else prints it either.
                self.error_traceback = traceback.format_exc()
        finally:
            self._remove_hooks()  # never leave forward hooks on the models
        if self.snapshot is not None:
            self.snapshot["finished"] = datetime.now().isoformat(timespec="seconds")
            self.snapshot["state"] = self.state
        self._emit_status()
        # Every terminal run joins the run history (weightless; see the store's
        # retention rules) — after the status emit so tabs settle state first.
        if self._record_runs:
            from . import checkpoints

            checkpoints.record(self)

    def _merged(self, live: dict[str, list[float]]) -> dict[str, list[float]]:
        """The checkpoint's stored history (when resuming) + the live run's —
        one continuous curve. A fresh run's base is empty, so this is a copy."""
        merged = {k: list(v) for k, v in self._base_history.items()}
        for k, v in live.items():
            merged[k] = merged.get(k, []) + list(v)
        return merged

    def _mid_run_checkpoint(self) -> dict[str, Any]:
        """A complete checkpoint cut at the current epoch boundary (autosave):
        CPU-cloned live weights + history-so-far + this run's snapshot — fully
        resumable under warm-start semantics, same shape as checkpoint()."""
        def clone(sd):
            return {k: v.detach().cpu().clone() for k, v in sd.items()}

        return {
            "version": CHECKPOINT_VERSION,
            "state_dicts": {role: clone(m.state_dict()) for role, m in self._live_models.items()},
            "best_state_dict": self.best_state_dict,
            "best_epoch": self.best_epoch,
            "epoch": self.epoch,
            "history": {k: list(v) for k, v in (self.history or {}).items()},
            "health_history": list(self._health_history),
            "snapshot": dict(self.snapshot) if self.snapshot else None,
        }

    def _register_activation_hooks(self, models: dict[str, Any]) -> None:
        """Hook the activation layers where a ~0 output genuinely means the unit
        isn't firing, so we can track dead units (units stuck at ~0 all epoch —
        a dead ReLU). Only activations whose *resting* state is zero qualify:
        ReLU/ReLU6 floor at 0, GELU/SiLU floor toward 0 (with vanishing gradient
        there). Deliberately NOT tanh/sigmoid/softmax/leaky/ELU — for those,
        output ≈ 0 isn't 'dead' (tanh's zero is its *active* region), so a dead-
        unit measure would mislead. These carry no params, so _collect_health's
        norm walk misses them; the hook fills the gap. Torn down at run end."""
        for role, model in models.items():
            submods = dict(model.named_modules())
            for layer, ln in self._layer_map.get(role, {}).items():
                if ln.type in _ZERO_FLOOR_ACTIVATIONS and (module := submods.get(layer)) is not None:
                    self._hook_handles.append(module.register_forward_hook(self._activation_hook(role, layer)))

    def _activation_hook(self, role: str, layer: str):
        import torch

        def hook(_module: Any, _inp: Any, output: Any) -> None:
            # Per-unit (dim 1 — channels for conv, features for linear) "activated
            # at all this forward", OR-ed into the epoch's alive mask.
            if not isinstance(output, torch.Tensor) or output.ndim < 2:
                return
            flat = output.detach().abs().transpose(0, 1).reshape(output.shape[1], -1)
            alive_now = (flat > 1e-6).any(dim=1).cpu()
            masks = self._alive_masks.setdefault(role, {})
            prev = masks.get(layer)
            masks[layer] = alive_now if prev is None else (prev | alive_now)

        return hook

    def _remove_hooks(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []

    def _collect_health(self) -> dict[str, dict[str, Any]]:
        """Cheap per-layer norms for each live model, keyed by canvas node: the
        weight L2 norm, the update ratio ‖Δw‖/‖w‖ vs. the previous epoch, and —
        best-effort, when grads are still present at the epoch boundary — the
        gradient norm. Parameters group by their `layer_N` prefix; layers without
        learnable params (activations) don't appear. Stashes this epoch's weights
        for the next delta. O(params) copy per epoch — negligible vs. a step."""
        import torch

        prev = self._prev_weights or {}
        new_prev: dict[str, dict[str, Any]] = {}
        snapshot: dict[str, dict[str, Any]] = {}
        for role, model in self._live_models.items():
            lmap = self._layer_map.get(role, {})
            by_layer: dict[str, list] = {}
            for pname, p in model.named_parameters():
                by_layer.setdefault(pname.split(".", 1)[0], []).append(p)
            role_stats: dict[str, dict[str, Any]] = {}
            role_prev: dict[str, Any] = {}
            for layer, params in by_layer.items():
                flat = torch.cat([p.detach().reshape(-1) for p in params]).float().cpu()
                ln = lmap.get(layer)
                stat: dict[str, Any] = {
                    "node": ln.label if ln else layer,
                    "nodeId": ln.node_id if ln else None,
                    "w": float(flat.norm()),
                }
                pv = prev.get(role, {}).get(layer)
                if pv is not None and pv.numel() == flat.numel():
                    stat["dw"] = float((flat - pv).norm()) / (float(pv.norm()) + 1e-12)
                grads = [p.grad for p in params]
                if all(g is not None for g in grads):
                    gflat = torch.cat([g.detach().reshape(-1) for g in grads]).float()
                    stat["g"] = float(gflat.norm())
                role_stats[layer] = stat
                role_prev[layer] = flat
            snapshot[role] = role_stats
            new_prev[role] = role_prev

        # Activation layers carry no params, so they got no row above — add one
        # from this epoch's dead-unit mask (fraction of units that never left ~0).
        for role, lmap in self._layer_map.items():
            role_stats = snapshot.setdefault(role, {})
            masks = self._alive_masks.get(role, {})
            for layer, ln in lmap.items():
                if layer in role_stats:  # already has a parametric row
                    continue
                mask = masks.get(layer)
                if mask is None:  # not an activation (or never ran)
                    continue
                role_stats[layer] = {
                    "node": ln.label,
                    "nodeId": ln.node_id,
                    "dead": float((~mask).float().mean()),
                }

        self._prev_weights = new_prev
        self._alive_masks = {}  # start the next epoch's dead-unit tracking fresh
        return snapshot

    def _on_epoch(self, epoch: int, history: dict[str, list[float]]) -> bool:
        """The generated train()'s per-epoch hook: record progress, capture the
        best-val weights, autosave, push to open tabs, and return False to
        request a cooperative stop. `epoch` counts the live run; the reported
        epoch adds the resume offset, so numbering continues across the seam."""
        # Wall time of this epoch (measured first, before the boundary work below).
        now = time.perf_counter()
        secs = now - self._last_epoch_ts
        self._last_epoch_ts = now

        # Order matters for the lock-free reader (see the class docstring):
        # publish the merged history BEFORE the epoch count, so a concurrent
        # status() never reports an epoch ahead of the curve it can show.
        self.history = self._merged(history)
        # Already absolute: the generated loop runs `range(start_epoch, epochs)`
        # over the whole plan, so it reports plan-relative numbers itself.
        # _epoch_offset survives for the two things that still need "how many
        # came before" — this segment's step budget and the step chart's x-axis.
        self.epoch = epoch

        # Per-layer health snapshot (weight/update/grad norms, keyed by node).
        # Reassigned as a whole list, lock-free like history above.
        health = self._collect_health()
        self._health_history = [*self._health_history, health]

        # New val_loss minimum → snapshot the weights NOW (CPU clones — the live
        # model keeps training, so a reference would silently drift).
        val = history.get("val_loss") or []
        if val:
            self._epochs_since_best = 0 if val[-1] < self._best_val else self._epochs_since_best + 1
        if val and val[-1] < self._best_val and self._live_model is not None:
            self._best_val = val[-1]
            self.best_epoch = self.epoch
            self.best_state_dict = {
                k: v.detach().cpu().clone() for k, v in self._live_model.state_dict().items()
            }

        # Periodic autosave: a rolling store entry, overwritten each interval
        # (single- or multi-model, whichever this run is).
        if self._autosave_every and self._live_models and epoch % self._autosave_every == 0:
            from .checkpoints import save_entry

            save_entry("autosave", self._mid_run_checkpoint())

        self._emit(
            {
                "type": "run_epoch",
                "epoch": self.epoch,
                "epochs": self.epochs,
                "metrics": {k: v[-1] for k, v in history.items() if v},
                "health": health,
                "secs": secs,
            }
        )
        # Early stop: val hasn't improved for `patience` epochs. The best-val
        # weights were captured as they happened, so stopping costs nothing;
        # the run ends "done" (a completion by criterion, not a user stop).
        early = (
            self._early_stop_patience > 0
            and self._epochs_since_best >= self._early_stop_patience
        )
        return not self._stop_requested and not early

    def _on_step(self, step: int, metrics: dict[str, float]) -> None:
        """The generated loop's per-batch hook: stream a throttled step-metrics
        point (``{name: value}`` — a single train_loss for supervised, or a GAN's
        g/d and a VAE's recon/kl) so intra-epoch loss is visible before the epoch
        ends. Time-throttled to keep the socket sane on fast loops; the first step
        of a run always emits (the clock is reset at run start)."""
        now = time.perf_counter()
        if now - self._last_step_emit < _STEP_EMIT_INTERVAL:
            return
        self._last_step_emit = now
        point = {"step": step, "metrics": {k: float(v) for k, v in metrics.items()}}
        # The point's position on the epoch axis, baked in AT BIRTH — the one
        # moment offset/steps_per_epoch are certainly this segment's own. A
        # resumed run's old points keep their baked x (from their checkpoint),
        # so old + new segments concatenate into one continuous curve; mapping
        # at render time with a single "current" span mislabels exactly them.
        if self._steps_per_epoch > 0:
            point["epoch_x"] = self._epoch_offset + step / self._steps_per_epoch
        # Keep the emitted point for late joiners. Reassigned as a whole (the
        # lock-free status() contract); at the cap, halve density instead of
        # sliding so the buffer always spans the run from step 1.
        history = self._step_history + [point]
        self._step_history = history[::2] if len(history) > _STEP_HISTORY_LIMIT else history
        self._emit({"type": "run_step", "total": self._total_steps, **point})

    def _emit_status(self) -> None:
        self._emit(
            {
                "type": "run_status",
                "state": self.state,
                "error": self.error,
                "epoch": self.epoch,
                "epochs": self.epochs,
                "seed": self.seed,
                "best_epoch": self.best_epoch,
                # The run's own recorded config, so live tabs can label the
                # dashboard with what actually ran (the form edits the NEXT run).
                # (No step_span here: "running" precedes the thread computing it —
                # the span rides run_step events and status(), never stale.)
                "config": self.run_config(),
                "run_name": self.run_name,
            }
        )


run_manager = RunManager(record_runs=True)

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
from datetime import datetime
from typing import Any, Callable

from .codegen import (
    class_name_for,
    exec_generated,
    generate_dataloader,
    generate_module,
    model_inputs,
)
from .datastore import registry
from .inference import build_incoming, graph_issues
from .introspect import variable_kind
from .recipes import get_recipe
from .registry import default_data, default_training
from .schema import Graph, Project, graph_from_project, project_from_graph, resolve_data_config


def _exec_source(source: str, wanted: str, filename: str) -> Any:
    """Run generated source (via the audited ``codegen.exec_generated``
    chokepoint) and return the named object — the same pattern the notebook
    client (build_model/build_trainer) uses."""
    return exec_generated(source, filename)[wanted]


def _exec_model(source: str, filename: str) -> Any:
    """Build the model class from generated module source, found by its type
    (the sole ``nn.Module`` subclass) rather than a fixed name — so a per-model
    class name (``Generator``, ``Discriminator``) works the same as the classic
    ``GeneratedModel``."""
    import torch.nn as nn

    for value in exec_generated(source, filename).values():
        if isinstance(value, type) and issubclass(value, nn.Module) and value is not nn.Module:
            return value
    raise ValueError("generated source defined no model class")


def _model_by_id(project: Project, model_id: str | None):
    return next((m for m in project.models if m.id == model_id), None)




class RunManager:
    """State machine for the single in-kernel training run."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_requested = False
        self.state: str = "idle"  # idle | running | done | stopped | failed
        self.error: str | None = None
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
        # Full reproducibility record of the current/last run: seed, resolved
        # device, effective configs, the graph, and the exact generated sources.
        self.snapshot: dict[str, Any] | None = None
        self._emit: Callable[[dict], None] = lambda message: None

    # -- public API ----------------------------------------------------------

    def start(
        self,
        design: Graph | Project,
        namespace: dict[str, Any] | None = None,
        emit: Callable[[dict], None] | None = None,
    ) -> str | None:
        """Validate and launch a run for a single graph or a whole project
        (multi-model, e.g. a GAN). Returns an error message if the run can't
        start (already running, invalid graph, unassigned role, unresolvable
        data), else None. Data and codegen are resolved *now*, so the thread
        never touches the namespace and a bad pick fails before anything starts."""
        ns = registry() if namespace is None else namespace
        if emit is None:
            from .ws import manager

            emit = manager.broadcast_threadsafe

        with self._lock:
            if self.state == "running":
                return "a run is already in progress — stop it first"

            project = design if isinstance(design, Project) else project_from_graph(design)
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

            # The data feeding the recipe's data-fed model (a GAN's discriminator,
            # the model for supervised): the dataset node wired into it.
            # needs_targets comes from the recipe.
            data_model_id = assignment.get(recipe.data_role) or (
                project.models[0].id if project.models else None
            )
            data_model = _model_by_id(project, data_model_id)
            data_config = resolve_data_config(project, data_model_id)
            data_graph = self._loader_graph(data_model.graph, project.links, data_model_id, data_config)
            try:
                call = self._resolve_call(data_graph, ns, needs_targets=recipe.needs_targets)
                # All codegen happens here, against the same namespace snapshot the
                # data was resolved from — the thread only execs sources. One
                # source per assigned model; the trainer comes from the recipe.
                sole = len(project.models) <= 1
                call["model_sources"] = {
                    role: generate_module(
                        m.graph, class_name=class_name_for(m.name, sole)
                    )
                    for role, mid in assignment.items()
                    if (m := _model_by_id(project, mid))
                }
                call["trainer_source"] = recipe.generate(project)
                call["data_source"] = generate_dataloader(
                    data_graph, namespace=ns, needs_targets=recipe.needs_targets
                )
            except ValueError as exc:
                return str(exc)

            cfg = {**{p.name: p.default for p in recipe.params}, **(project.training or {})}
            # Resolve the run's seed now so the snapshot is complete at start:
            # an unset seed is drawn at random AND recorded, so every run stays
            # reproducible. The thread applies it before anything touches RNG.
            seed = cfg.get("seed")
            call["seed"] = random.randrange(2**31) if seed is None else int(seed)
            device = str(cfg.get("device", "auto"))
            if device == "auto":
                from .registry import available_devices

                av = available_devices()
                device = "cuda" if "cuda" in av else "mps" if "mps" in av else "cpu"

            self.state = "running"
            self.error = None
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
            self._base_history = {}
            self._autosave_every = int(cfg.get("autosave_every") or 0)
            call["recipe"] = recipe.name
            self.snapshot = self._build_snapshot(project, assignment, cfg, device, call, data_config)
            self._stop_requested = False
            self._emit = emit
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
        return assignment, None

    def _build_snapshot(
        self, project: Project, assignment: dict[str, str], cfg: dict, device: str,
        call: dict, data_config: dict,
    ) -> dict[str, Any]:
        """The run's reproducibility record. ``data`` is the RESOLVED data config
        (the wired dataset node's, or the Data form) so a resume rebuilds the same
        loader. A single-model run keeps the v2 shape (``graph`` + ``sources.model``)
        so existing checkpoints/tests are untouched; a multi-model run records the
        whole ``project`` and per-role ``sources.models``."""
        base: dict[str, Any] = {
            "seed": call["seed"],
            "device": device,
            "training": cfg,
            "data": {**default_data(), **data_config},
            "started": datetime.now().isoformat(timespec="seconds"),
        }
        if len(assignment) == 1:
            role = next(iter(assignment))
            # The model graph carries the resolved data config, so resume rebuilds
            # the same loader whether data came from a node or the form.
            base["graph"] = graph_from_project(project).model_copy(update={"data": dict(data_config)}).model_dump()
            base["sources"] = {
                "model": call["model_sources"][role],
                "data": call["data_source"],
                "trainer": call["trainer_source"],
            }
        else:
            base["project"] = project.model_dump()
            base["sources"] = {
                "models": call["model_sources"],
                "data": call["data_source"],
                "trainer": call["trainer_source"],
            }
        return base

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

        with self._lock:
            if self.state == "running":
                return "a run is already in progress — stop it first"

            snapshot = checkpoint["snapshot"]
            multi = "project" in snapshot
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

            # Rebuild the design (a single-model snapshot stores a graph, a
            # multi-model one the whole project). The model source(s) travel
            # verbatim (the weights match them); the trainer is regenerated with
            # the REMAINING count baked in (a stored trainer bakes its own run's
            # count, rarely what's left). Data codegen re-runs against the current
            # namespace so repointed names keep working.
            if multi:
                project = Project.model_validate(snapshot["project"])
                model_sources = snapshot["sources"]["models"]
            else:
                project = project_from_graph(Graph.model_validate(snapshot["graph"]))
                model_sources = {"model": snapshot["sources"]["model"]}
            # The loader is built from the recipe's data-fed model (minus any label
            # port — see _loader_graph), exactly as `start` does, so a conditional
            # resume yields (X, y) too. The snapshot's RESOLVED data config comes
            # from snapshot.data (multi) or the single-model graph's own data.
            roles = (project.training or {}).get("roles") or {}
            data_model_id = roles.get(recipe.data_role) or project.models[0].id
            data_model = _model_by_id(project, data_model_id)
            data_config = dict(snapshot.get("data") or {}) if multi else resolve_data_config(project, data_model_id)
            data_graph = self._loader_graph(data_model.graph, project.links, data_model_id, data_config)
            try:
                call = self._resolve_call(data_graph, ns, needs_targets=recipe.needs_targets)
                call["model_sources"] = model_sources
                call["recipe"] = recipe.name
                gen_project = project.model_copy(deep=True)
                gen_project.training = {**(project.training or {}), "epochs": remaining}
                call["trainer_source"] = recipe.generate(gen_project)
                call["data_source"] = generate_dataloader(
                    data_graph, namespace=ns, needs_targets=recipe.needs_targets
                )
            except ValueError as exc:
                return str(exc)

            if multi:
                call["state_dicts"] = checkpoint["state_dicts"]
            else:
                call["state_dict"] = checkpoint["state_dict"]
            # A warm start is a NEW run: fresh drawn seed, recorded like any
            # other (the stored seed already had its run).
            call["seed"] = random.randrange(2**31)
            cfg["seed"] = call["seed"]

            self.state = "running"
            self.error = None
            self.epoch = offset
            self.epochs = target
            self.seed = call["seed"]
            self.model = None
            self.models = {}
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
            design = {"project": snapshot["project"]} if multi else {"graph": snapshot["graph"]}
            sources = {"data": call["data_source"], "trainer": call["trainer_source"]}
            sources["models" if multi else "model"] = model_sources if multi else model_sources["model"]
            self.snapshot = {
                "seed": call["seed"],
                "device": snapshot["device"],  # resolved when the original run started
                "training": cfg,
                "data": snapshot["data"],
                **design,
                "sources": sources,
                "started": datetime.now().isoformat(timespec="seconds"),
                "resumed_from": name,
                "resumed_at_epoch": offset,
            }
            self._stop_requested = False
            self._emit = emit
            self._emit_status()
            self._thread = threading.Thread(
                target=self._run, args=(call,), daemon=True, name="lamplighter-run"
            )
            self._thread.start()
        return None

    def stop(self) -> None:
        """Request a cooperative stop — honored at the next epoch boundary."""
        self._stop_requested = True

    def status(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "error": self.error,
            "epoch": self.epoch,
            "epochs": self.epochs,
            "seed": self.seed,
            "best_epoch": self.best_epoch,
            "history": self.history,
        }

    def join(self, timeout: float | None = None) -> bool:
        """Wait for the current run's thread (tests). True if it finished."""
        t = self._thread
        if t is None:
            return True
        t.join(timeout)
        return not t.is_alive()

    def checkpoint(self) -> dict[str, Any]:
        """The trained weights + the run snapshot, as one torch-saveable dict.
        Self-contained: the snapshot carries the generated model source(s), so the
        checkpoint can be rebuilt anywhere via lamplighter.load_checkpoint().
        Carries the best-val weights too, when validation ran. A single-model run
        writes the v2 shape (``state_dict``); a multi-model run writes v3
        (``state_dicts`` keyed by role)."""
        if not self.models:
            raise ValueError("no trained model yet — run training first")
        if len(self.models) == 1:
            return {
                "state_dict": self.model.state_dict(),
                "best_state_dict": self.best_state_dict,
                "best_epoch": self.best_epoch,
                "epoch": len((self.history or {}).get("train_loss", [])),
                "history": self.history,
                "snapshot": self.snapshot,
            }
        return {
            "state_dicts": {role: m.state_dict() for role, m in self.models.items()},
            "best_state_dict": None,  # best-epoch tracking is a has_val (supervised) feature
            "best_epoch": None,
            "epoch": max((len(v) for v in (self.history or {}).values()), default=0),
            "history": self.history,
            "snapshot": self.snapshot,
        }

    def restore(self, checkpoint: dict[str, Any]) -> str | None:
        """Repopulate the run artifacts from a stored checkpoint: the model is
        rebuilt from the checkpoint's own generated source + final weights, so
        sess.model, the weights download, and resume all behave as if that run
        had just finished. Returns an error message if refused (mid-run), else
        None. load_state_dict copies the weights in, so the store's entry stays
        isolated from whatever happens to the live model afterwards."""
        with self._lock:
            if self.state == "running":
                return "a run is in progress — stop it before restoring a checkpoint"
            snapshot = checkpoint["snapshot"]
            history = checkpoint.get("history") or {}

            if "state_dicts" in checkpoint:  # v3 multi-model
                sources = snapshot["sources"]["models"]
                models: dict[str, Any] = {}
                for role, sd in checkpoint["state_dicts"].items():
                    cls = _exec_model(sources[role], f"<lamplighter-restore-{role}>")
                    m = cls()
                    m.load_state_dict(sd)
                    models[role] = m.eval()
                self.models = models
                self.model = None
                self.best_epoch = None
                self.best_state_dict = None
                self._best_val = float("inf")
            else:  # v2 single-model
                cls = _exec_model(
                    snapshot["sources"]["model"], "<lamplighter-restore-model>"
                )
                model = cls()
                model.load_state_dict(checkpoint["state_dict"])
                self.models = {"model": model.eval()}
                self.model = model
                self.best_epoch = checkpoint.get("best_epoch")
                self.best_state_dict = checkpoint.get("best_state_dict")
                val = history.get("val_loss") or []
                self._best_val = min(val) if val else float("inf")

            self.state = "done"
            self.error = None
            self.epochs = checkpoint.get("epoch")
            if self.epochs is None:
                self.epochs = max((len(v) for v in history.values()), default=0)
            self.epoch = self.epochs
            self.seed = snapshot.get("seed")
            self.history = {k: list(v) for k, v in history.items()} or None
            self.snapshot = snapshot
        return None

    def best_model(self) -> Any:
        """Rebuild the best-val-epoch model from the run's own generated source
        (None when validation didn't run). Fresh instance, eval mode, CPU."""
        if self.best_state_dict is None or self.snapshot is None:
            return None
        model_cls = _exec_model(
            self.snapshot["sources"]["model"], "<lamplighter-best-model>"
        )
        model = model_cls()
        model.load_state_dict(self.best_state_dict)
        return model.eval()

    # -- data resolution (pre-flight) -----------------------------------------

    def _resolve_call(
        self, graph: Graph, ns: dict[str, Any], needs_targets: bool = True
    ) -> dict[str, Any]:
        """Resolve the arguments the generated make_dataloaders() needs from the
        notebook namespace (all data flows through it — one path). Returns a call
        spec consumed by the thread body; raises ValueError with a user-facing
        message otherwise. ``needs_targets=False`` (an adversarial recipe)
        resolves the input X alone — no target."""
        data = {**default_data(), **(graph.data or {})}
        source = str(data["source"])  # "memory" | "torchvision" | "imagefolder"

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
    def _loader_graph(base: Graph, links, data_model_id, data_config: dict) -> Graph:
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
            return Graph(nodes=base.nodes, edges=base.edges, data=data_config)
        nodes = [n for n in base.nodes if n.id not in label_ids]
        edges = [e for e in base.edges if e.source not in label_ids and e.target not in label_ids]
        return Graph(nodes=nodes, edges=edges, data=data_config)

    def _resolve_inputs(self, graph: Graph, data: dict, ns: dict[str, Any]) -> list[Any]:
        """The picked input variable(s), ordered to match forward() args. Single
        input uses data.x_var; multi-input uses data.x_vars keyed by Input node
        id, ordered by canvas position at run time (robust to reordering)."""
        node_map = {n.id: n for n in graph.nodes}
        incoming = build_incoming(graph)
        input_ids = model_inputs(graph, incoming, node_map)

        if len(input_ids) <= 1:
            return [self._resolve_tensor(data.get("x_var"), "input (X)", ns)]

        x_vars = data.get("x_vars") or {}
        xs: list[Any] = []
        for i, nid in enumerate(input_ids):
            name = str(x_vars.get(nid, "") or "").strip()
            label = node_map[nid].params.get("name") or f"Input {i}"
            if not name:
                raise ValueError(f"no variable picked for {label} — pick one in the Data tab")
            xs.append(self._resolve_tensor(name, str(label), ns))
        return xs

    @staticmethod
    def _resolve_tensor(name: Any, what: str, ns: dict[str, Any]) -> Any:
        """A named notebook variable that must be a torch.Tensor."""
        name = str(name or "").strip()
        if not name:
            raise ValueError(f"no variable picked for the {what} — pick one in the Data tab")
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
                # Warm start (resume): load stored weights — single via
                # state_dict, multi-model via state_dicts keyed by role.
                if call.get("state_dict") is not None and len(models) == 1:
                    next(iter(models.values())).load_state_dict(call["state_dict"])
                for role, sd in (call.get("state_dicts") or {}).items():
                    models[role].load_state_dict(sd)
                self._live_models = models
                # Single-model convenience for best-val capture / autosave; None
                # for a multi-model run (no per-role best tracking yet).
                self._live_model = next(iter(models.values())) if len(models) == 1 else None
                train = _exec_source(call["trainer_source"], "train", "<lamplighter-run-trainer>")
                make = _exec_source(call["data_source"], "make_dataloaders", "<lamplighter-run-data>")
                train_loader, val_loader = make(*call["loader_args"])
                recipe = get_recipe(call["recipe"])
                history = recipe.bind(train, models, train_loader, val_loader, self._on_epoch)

            with self._lock:
                self.models = models
                self.model = next(iter(models.values())) if len(models) == 1 else None
                self.history = self._merged(history)
                self.state = "stopped" if self._stop_requested else "done"
        except Exception as exc:  # surface anything from user data/generated code
            with self._lock:
                self.state = "failed"
                self.error = f"{type(exc).__name__}: {exc}"
        if self.snapshot is not None:
            self.snapshot["finished"] = datetime.now().isoformat(timespec="seconds")
            self.snapshot["state"] = self.state
        self._emit_status()

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
        resumable under warm-start semantics, same format as checkpoint() (v2 for
        a single model, v3 for several)."""
        def clone(sd):
            return {k: v.detach().cpu().clone() for k, v in sd.items()}

        base = {
            "epoch": self.epoch,
            "history": {k: list(v) for k, v in (self.history or {}).items()},
            "snapshot": dict(self.snapshot) if self.snapshot else None,
        }
        if len(self._live_models) == 1:
            return {
                "state_dict": clone(next(iter(self._live_models.values())).state_dict()),
                "best_state_dict": self.best_state_dict,
                "best_epoch": self.best_epoch,
                **base,
            }
        return {
            "state_dicts": {role: clone(m.state_dict()) for role, m in self._live_models.items()},
            "best_state_dict": None,
            "best_epoch": None,
            **base,
        }

    def _on_epoch(self, epoch: int, history: dict[str, list[float]]) -> bool:
        """The generated train()'s per-epoch hook: record progress, capture the
        best-val weights, autosave, push to open tabs, and return False to
        request a cooperative stop. `epoch` counts the live run; the reported
        epoch adds the resume offset, so numbering continues across the seam."""
        self.epoch = epoch + self._epoch_offset
        # Keep the merged history current, so late-joining tabs (and autosaves)
        # see the whole curve mid-run.
        self.history = self._merged(history)

        # New val_loss minimum → snapshot the weights NOW (CPU clones — the live
        # model keeps training, so a reference would silently drift).
        val = history.get("val_loss") or []
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
            }
        )
        return not self._stop_requested

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
            }
        )


run_manager = RunManager()

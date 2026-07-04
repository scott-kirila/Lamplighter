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
    generate_dataloader,
    generate_module,
    model_inputs,
)
from .datastore import registry
from .inference import build_incoming, graph_issues
from .introspect import variable_kind
from .recipes import get_recipe
from .registry import default_data, default_training
from .schema import Graph, project_from_graph


def _exec_source(source: str, wanted: str, filename: str) -> Any:
    """exec generated source in a fresh namespace and return the named object —
    the same pattern the notebook client (build_model/build_trainer) uses."""
    ns: dict[str, Any] = {}
    exec(compile(source, filename, "exec"), ns)  # noqa: S102
    return ns[wanted]


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
        self.model: Any = None
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
        graph: Graph,
        namespace: dict[str, Any] | None = None,
        emit: Callable[[dict], None] | None = None,
    ) -> str | None:
        """Validate and launch a run. Returns an error message if the run can't
        start (already running, invalid graph, unresolvable data), else None.
        Data variables are resolved *now*, so the thread never touches the
        namespace and a bad pick fails before anything starts."""
        ns = registry() if namespace is None else namespace
        if emit is None:
            from .ws import manager

            emit = manager.broadcast_threadsafe

        with self._lock:
            if self.state == "running":
                return "a run is already in progress — stop it first"

            issues = graph_issues(graph)
            if issues:
                return "; ".join(issues)
            recipe = get_recipe((graph.training or {}).get("recipe"))
            if recipe is None:
                return f"unknown training recipe '{(graph.training or {}).get('recipe')}'"
            try:
                call = self._resolve_call(graph, ns)
                # All codegen happens here, against the same namespace snapshot
                # the data was resolved from — the thread only execs sources, so
                # what runs can't diverge from what was validated (or shown). The
                # trainer comes from the selected recipe (supervised = the classic
                # loop, byte-identical).
                call["model_source"] = generate_module(graph)
                call["trainer_source"] = recipe.generate(project_from_graph(graph))
                call["data_source"] = generate_dataloader(graph, namespace=ns)
            except ValueError as exc:
                return str(exc)

            cfg = {**default_training(), **(graph.training or {})}
            # Resolve the run's seed now so the snapshot is complete at start:
            # an unset seed is drawn at random AND recorded, so every run stays
            # reproducible. The thread applies it before anything touches RNG.
            seed = cfg.get("seed")
            call["seed"] = random.randrange(2**31) if seed is None else int(seed)
            device = str(cfg["device"])
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
            self.history = None
            self.best_epoch = None
            self.best_state_dict = None
            self._best_val = float("inf")
            self._epoch_offset = 0
            self._base_history = {}
            self._autosave_every = int(cfg.get("autosave_every") or 0)
            self.snapshot = {
                "seed": call["seed"],
                "device": device,
                "training": cfg,
                "data": {**default_data(), **(graph.data or {})},
                "graph": graph.model_dump(),
                "sources": {
                    "model": call["model_source"],
                    "data": call["data_source"],
                    "trainer": call["trainer_source"],
                },
                "started": datetime.now().isoformat(timespec="seconds"),
            }
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
            graph = Graph.model_validate(snapshot["graph"])
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
            recipe = get_recipe((graph.training or {}).get("recipe"))
            if recipe is None:
                return f"unknown training recipe '{(graph.training or {}).get('recipe')}'"
            try:
                call = self._resolve_call(graph, ns)
                # The model source travels verbatim (the weights match it). The
                # trainer is regenerated from the stored graph with the REMAINING
                # count baked in (a stored trainer bakes its own run's count, which
                # is rarely what's left to train), through the recipe. Data codegen
                # re-runs against the current namespace so repointed names keep working.
                call["model_source"] = snapshot["sources"]["model"]
                graph.training = {**(graph.training or {}), "epochs": remaining}
                call["trainer_source"] = recipe.generate(project_from_graph(graph))
                call["data_source"] = generate_dataloader(graph, namespace=ns)
            except ValueError as exc:
                return str(exc)

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
            self._epoch_offset = offset
            self._base_history = {
                k: list(v) for k, v in (checkpoint.get("history") or {}).items()
            }
            self.history = dict(self._base_history) or None
            # Best-so-far carries across the seam: the resumed run only claims
            # the marker by actually beating the stored minimum.
            self.best_epoch = checkpoint.get("best_epoch")
            self.best_state_dict = checkpoint.get("best_state_dict")
            val = self._base_history.get("val_loss") or []
            self._best_val = min(val) if val else float("inf")
            self._autosave_every = int(cfg.get("autosave_every") or 0)
            self.snapshot = {
                "seed": call["seed"],
                "device": snapshot["device"],  # resolved when the original run started
                "training": cfg,
                "data": snapshot["data"],
                "graph": snapshot["graph"],
                "sources": {
                    "model": call["model_source"],
                    "data": call["data_source"],
                    "trainer": call["trainer_source"],
                },
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
        Self-contained: the snapshot carries the generated model source, so the
        checkpoint can be rebuilt anywhere via lamplighter.load_checkpoint().
        Carries the best-val weights too, when validation ran."""
        if self.model is None:
            raise ValueError("no trained model yet — run training first")
        return {
            "state_dict": self.model.state_dict(),
            "best_state_dict": self.best_state_dict,
            "best_epoch": self.best_epoch,
            "epoch": len((self.history or {}).get("train_loss", [])),
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
            model_cls = _exec_source(
                snapshot["sources"]["model"], "GeneratedModel", "<lamplighter-restore-model>"
            )
            model = model_cls()
            model.load_state_dict(checkpoint["state_dict"])
            history = checkpoint.get("history") or {}

            self.state = "done"
            self.error = None
            self.epochs = checkpoint.get("epoch") or len(history.get("train_loss", []))
            self.epoch = self.epochs
            self.seed = snapshot.get("seed")
            self.model = model.eval()
            self.history = {k: list(v) for k, v in history.items()} or None
            self.best_epoch = checkpoint.get("best_epoch")
            self.best_state_dict = checkpoint.get("best_state_dict")
            val = history.get("val_loss") or []
            self._best_val = min(val) if val else float("inf")
            self.snapshot = snapshot
        return None

    def best_model(self) -> Any:
        """Rebuild the best-val-epoch model from the run's own generated source
        (None when validation didn't run). Fresh instance, eval mode, CPU."""
        if self.best_state_dict is None or self.snapshot is None:
            return None
        model_cls = _exec_source(
            self.snapshot["sources"]["model"], "GeneratedModel", "<lamplighter-best-model>"
        )
        model = model_cls()
        model.load_state_dict(self.best_state_dict)
        return model.eval()

    # -- data resolution (pre-flight) -----------------------------------------

    def _resolve_call(self, graph: Graph, ns: dict[str, Any]) -> dict[str, Any]:
        """Resolve the arguments the generated make_dataloaders() needs from the
        notebook namespace (all data flows through it — one path). Returns a call
        spec consumed by the thread body; raises ValueError with a user-facing
        message otherwise."""
        data = {**default_data(), **(graph.data or {})}
        source = str(data["source"])  # "memory" | "torchvision" | "imagefolder"

        if source == "memory":
            x_var = str(data.get("x_var", "") or "").strip()
            kind = variable_kind(x_var, ns) if x_var else None
            if kind in ("dataloader", "dataset"):
                return {"loader_args": (ns[x_var],)}
            xs = self._resolve_inputs(graph, data, ns)
            y = self._resolve_tensor(data.get("y_var"), "target (y)", ns)
            return {"loader_args": (*xs, y)}
        # torchvision / imagefolder need no data arguments (may download in-run).
        return {"loader_args": ()}

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

                model_cls = _exec_source(
                    call["model_source"], "GeneratedModel", "<lamplighter-run-model>"
                )
                model = model_cls()
                if call.get("state_dict") is not None:  # warm start (resume)
                    model.load_state_dict(call["state_dict"])
                self._live_model = model  # visible to _on_epoch for best-val capture
                train = _exec_source(call["trainer_source"], "train", "<lamplighter-run-trainer>")
                make = _exec_source(call["data_source"], "make_dataloaders", "<lamplighter-run-data>")
                train_loader, val_loader = make(*call["loader_args"])
                history = train(
                    model, train_loader, val_loader=val_loader, on_epoch=self._on_epoch
                )

            with self._lock:
                self.model = model
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
        resumable under warm-start semantics, same format as checkpoint()."""
        return {
            "state_dict": {
                k: v.detach().cpu().clone() for k, v in self._live_model.state_dict().items()
            },
            "best_state_dict": self.best_state_dict,
            "best_epoch": self.best_epoch,
            "epoch": self.epoch,
            "history": {k: list(v) for k, v in (self.history or {}).items()},
            "snapshot": dict(self.snapshot) if self.snapshot else None,
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

        # Periodic autosave: a rolling store entry, overwritten each interval.
        if self._autosave_every and epoch % self._autosave_every == 0:
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

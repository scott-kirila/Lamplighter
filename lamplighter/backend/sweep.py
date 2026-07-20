"""The Optimize engine: Optuna-driven hyperparameter sweeps as real runs.

A sweep is N sequential trials, each a REAL managed run: the trial's suggested
params are dict-merged into ``project.training`` and the patched project goes
through the ordinary :class:`RunManager` — so every trial auto-records into the
run store (source ``"sweep"``, tagged with its study), streams to open tabs
like any run, and carries a full reproducibility snapshot showing ITS OWN
hyperparameters. No parallel bookkeeping: the trials table is derived by
filtering the run listing on the study tag.

Pruning rides the existing machinery: the trial's per-epoch metric is reported
through a composed emit hook, and ``should_prune`` triggers the runner's
cooperative ``stop()`` — a pruned trial records as a normal "stopped" run.
The best COMPLETE trial's weights are kept (``checkpoints.save``) as they
happen — the kernel holds each trial's model right after it finishes, and only
then — so the winner is restorable/resumable when the sweep ends, at which
point it is renamed ``<study>-best`` (naming exempts it from retention; the
other trials rotate out as ordinary autos, with Optuna's study as the full
record).

Optuna is an OPTIONAL dependency (``pip install "lamplighter[sweep]"``),
imported lazily — nothing in this module executes at import time without it.
"""
from __future__ import annotations

import re
import threading
from typing import Any, Callable

from .schema import Project

# One trial param spec: {"name", "type": "float"|"int"|"categorical",
# float/int: "low"/"high" (+ "log" for float), categorical: "choices"}.
# v1 targets project.training keys; node params (hidden dims) are Phase C.
_PARAM_TYPES = ("float", "int", "categorical")


def _optuna():
    """Lazy import with the install hint — the tab shows this message verbatim."""
    try:
        import optuna
    except ImportError:
        raise ValueError(
            "Optuna isn't installed — hyperparameter sweeps need it: "
            "pip install \"lamplighter[sweep]\""
        ) from None
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna


def _check_config(config: dict[str, Any]) -> str | None:
    """A user-facing message for a malformed sweep config, or None. Validation
    the ▶ button needs before anything starts; per-trial validity (does the
    graph build, does the data resolve) stays the runner's job."""
    params = config.get("params") or []
    if not params:
        return "pick at least one hyperparameter to sweep"
    for p in params:
        name, ptype = str(p.get("name", "") or ""), str(p.get("type", "") or "")
        if not name:
            return "every swept param needs a name"
        if ptype not in _PARAM_TYPES:
            return f"param '{name}': unknown type '{ptype}' (float, int, or categorical)"
        if ptype == "categorical":
            if not p.get("choices"):
                return f"param '{name}': categorical needs choices"
        elif p.get("low") is None or p.get("high") is None:
            return f"param '{name}': needs low and high"
    n = config.get("n_trials")
    if not isinstance(n, int) or n < 1:
        return "n_trials must be a positive integer"
    if str(config.get("direction", "minimize")) not in ("minimize", "maximize"):
        return "direction must be minimize or maximize"
    return None


def _suggest(trial, spec: dict[str, Any]) -> Any:
    name, ptype = str(spec["name"]), str(spec["type"])
    if ptype == "float":
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]),
                                   log=bool(spec.get("log", False)))
    if ptype == "int":
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
    return trial.suggest_categorical(name, list(spec["choices"]))


class SweepManager:
    """State machine for the single in-kernel sweep (mirrors RunManager's
    shape). One sweep at a time; trials are SEQUENTIAL — the run manager is a
    singleton and that's the honest capacity of one kernel. Lifecycle
    transitions hold the lock; the sweep thread reassigns whole status fields
    (the run manager's lock-free reader contract)."""

    def __init__(self, manager: Any = None, emit: Callable[[dict], None] | None = None) -> None:
        # Injectable for tests: the run manager trials go through, and the
        # event sink (defaults to the WS broadcast, resolved lazily).
        self._manager = manager
        self._emit_override = emit
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_requested = False
        self._pruned_current = False
        self._prune_enabled = True
        self._live_trial: Any = None

        self.state: str = "idle"  # idle | running | done | stopped | failed
        self.error: str | None = None
        self.study_name: str | None = None
        self.n_trials = 0
        self.trial: int | None = None  # 1-based index of the running trial
        self.completed = 0
        self.pruned = 0
        self.failed = 0
        self.metric = "val_loss"
        self.direction = "minimize"
        self.best: dict[str, Any] | None = None  # {"run_name", "value", "params"}

    # -- plumbing -------------------------------------------------------------

    def _run_manager(self):
        if self._manager is not None:
            return self._manager
        from .runner import run_manager

        return run_manager

    def _emit(self, message: dict) -> None:
        if self._emit_override is not None:
            self._emit_override(message)
            return
        try:
            from .ws import manager

            manager.broadcast_threadsafe(message)
        except Exception:
            pass

    def _run_emit(self, message: dict) -> None:
        """The per-trial run emit: forward everything to the normal sink (tabs
        see trials stream exactly like hand-started runs), and watch run_epoch
        for the prune decision."""
        self._emit(message)
        if message.get("type") == "run_epoch":
            self._maybe_prune(message.get("epoch"), message.get("metrics") or {})

    def _maybe_prune(self, epoch: Any, metrics: dict[str, Any]) -> None:
        """Report this epoch's metric to the live trial; stop the run if the
        pruner says so. Runs on the TRAINING thread (emit's caller) — never
        raise into it."""
        trial = self._live_trial
        if trial is None or not self._prune_enabled:
            return
        value = metrics.get(self.metric)
        if value is None or epoch is None:
            return
        try:
            trial.report(float(value), step=int(epoch))
            if trial.should_prune():
                self._pruned_current = True
                self._run_manager().stop()
        except Exception:
            pass  # a pruner hiccup must not kill the training thread

    def _emit_status(self) -> None:
        self._emit({"type": "sweep_status", **self.status()})

    # -- public API -----------------------------------------------------------

    def start(self, project: Project, config: dict[str, Any], pruner: Any = None) -> str | None:
        """Validate and launch a sweep. Returns a user-facing error message
        (bad config, Optuna missing, a sweep or run already in progress), else
        None. ``pruner`` overrides the default MedianPruner (tests inject a
        deterministic one)."""
        err = _check_config(config)
        if err is not None:
            return err
        try:
            optuna = _optuna()
        except ValueError as exc:
            return str(exc)

        with self._lock:
            if self.state == "running":
                return "a sweep is already in progress — stop it first"
            if self._run_manager().state == "running":
                return "a run is in progress — stop it before starting a sweep"

            self.state = "running"
            self.error = None
            self.study_name = str(config.get("study") or self._next_study_name())
            self.n_trials = int(config["n_trials"])
            self.trial = None
            self.completed = 0
            self.pruned = 0
            self.failed = 0
            self.best = None
            self.metric = str(config.get("metric") or "val_loss")
            self.direction = str(config.get("direction") or "minimize")
            self._prune_enabled = bool(config.get("prune", True))
            self._live_trial = None
            self._stop_requested = False

            # Seeded sampler → the SWEEP itself is reproducible (each trial
            # still draws its own recorded training seed unless the project
            # pins one). MedianPruner needs a completed baseline before it
            # prunes, so trial 1 always finishes.
            sampler = optuna.samplers.TPESampler(seed=config.get("seed"))
            if pruner is None:
                pruner = (
                    optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=0)
                    if self._prune_enabled
                    else optuna.pruners.NopPruner()
                )
            study = optuna.create_study(direction=self.direction, sampler=sampler, pruner=pruner)

            self._emit_status()
            self._thread = threading.Thread(
                target=self._run_sweep,
                args=(optuna, study, project, list(config["params"])),
                daemon=True,
                name="lamplighter-sweep",
            )
            self._thread.start()
        return None

    def stop(self) -> None:
        """Cooperative stop: no new trials start, and the current trial's run
        is stopped too (so stopping doesn't wait out a whole trial)."""
        self._stop_requested = True
        self._run_manager().stop()

    def join(self, timeout: float | None = None) -> bool:
        t = self._thread
        if t is None:
            return True
        t.join(timeout)
        return not t.is_alive()

    def status(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "error": self.error,
            "study": self.study_name,
            "n_trials": self.n_trials,
            "trial": self.trial,
            "completed": self.completed,
            "pruned": self.pruned,
            "failed": self.failed,
            "metric": self.metric,
            "direction": self.direction,
            "best": self.best,
        }

    # -- the sweep itself (background thread) ---------------------------------

    def _run_sweep(self, optuna, study, project: Project, specs: list[dict]) -> None:
        manager = self._run_manager()
        try:
            for i in range(self.n_trials):
                if self._stop_requested:
                    break
                self.trial = i + 1
                trial = study.ask()
                params = {spec["name"]: _suggest(trial, spec) for spec in specs}

                patched = project.model_copy(deep=True)
                patched.training = {**(project.training or {}), **params}

                self._pruned_current = False
                self._live_trial = trial
                err = manager.start(
                    patched, emit=self._run_emit, source="sweep", study=self.study_name
                )
                if err is not None:
                    # A start refusal (invalid graph, bad data pick) would fail
                    # every trial identically — abort the sweep with the reason.
                    raise RuntimeError(err)
                manager.join()
                self._live_trial = None

                if self._pruned_current:
                    self.pruned += 1
                    study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                elif manager.state == "done":
                    value = self._trial_value(manager)
                    self.completed += 1
                    study.tell(trial, value)
                    self._maybe_new_best(manager, value, params)
                else:
                    # failed, or user-stopped mid-trial: the trial is data
                    # (recorded like any run) but yields no value.
                    self.failed += 1
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                self.trial = None
                self._emit_status()

            self._finish_best()
            self.state = "stopped" if self._stop_requested else "done"
        except Exception as exc:
            self.state = "failed"
            self.error = f"{type(exc).__name__}: {exc}" if not isinstance(exc, RuntimeError) else str(exc)
        finally:
            self._live_trial = None
            self.trial = None
        self._emit_status()

    def _trial_value(self, manager) -> float:
        """The completed trial's objective — the metric's final value. A missing
        metric fails the SWEEP fast with the fix spelled out (every trial would
        be missing it identically)."""
        series = (manager.history or {}).get(self.metric) or []
        if not series:
            have = ", ".join(sorted(manager.history or {})) or "nothing"
            raise RuntimeError(
                f"the sweep metric '{self.metric}' isn't in the run history "
                f"(recorded: {have}) — set a validation split, or sweep on train_loss"
            )
        return float(series[-1])

    def _maybe_new_best(self, manager, value: float, params: dict[str, Any]) -> None:
        """Keep the best trial's weights AS IT HAPPENS — the kernel holds this
        trial's model only until the next trial starts, so this is the one
        moment its weights can be saved. Saving upgrades the trial's auto
        record in place (and clears its auto flag → exempt from retention)."""
        better = self.best is None or (
            value < self.best["value"] if self.direction == "minimize" else value > self.best["value"]
        )
        if not better:
            return
        run_name = manager.run_name
        if run_name is not None:
            try:
                from . import checkpoints

                checkpoints.save(run_name, manager=manager)
            except ValueError:
                pass  # weightless edge (shouldn't happen on "done") — keep the record
        self.best = {"run_name": run_name, "value": value, "params": dict(params)}

    def _finish_best(self) -> None:
        """Rename the winner ``<study>-best`` so it reads as the sweep's
        artifact (named + weighted → permanently kept); a taken name gets a
        numeric suffix rather than clobbering a previous sweep's winner."""
        if self.best is None or self.best.get("run_name") is None:
            return
        from . import checkpoints

        base = f"{self.study_name}-best"
        name = base
        n = 2
        while True:
            try:
                checkpoints.rename(self.best["run_name"], name)
                self.best["run_name"] = name
                return
            except ValueError as exc:
                if "already exists" in str(exc):
                    name = f"{base}-{n}"
                    n += 1
                    continue
                return  # renamed under our feet (deleted?) — keep the old name

    def _next_study_name(self) -> str:
        """sweep-N, past the highest existing study number in the run store."""
        from . import checkpoints

        n = 0
        for meta in checkpoints.metas():
            m = re.fullmatch(r"sweep-(\d+)", str(meta.get("study") or ""))
            if m:
                n = max(n, int(m.group(1)))
        return f"sweep-{n + 1}"


sweep_manager = SweepManager()

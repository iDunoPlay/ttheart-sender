"""Executes parsed flows.

Responsibilities kept here rather than in individual actions: retries, optional
steps, the stop key, nesting depth for readable logs, and turning soft failures
into a flow-level verdict.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from ..exceptions import ActionError, FlowAborted, StopRequested, TTHeartError
from .context import RunContext
from .flow import Flow, Step
from .params import Params
from .registry import ActionResult

log = logging.getLogger(__name__)


@dataclass
class RunReport:
    flow: str
    success: bool = True
    steps_run: int = 0
    steps_failed: int = 0
    steps_skipped: int = 0
    duration: float = 0.0
    stopped_early: bool = False
    message: str = ""

    def summary(self) -> str:
        verdict = "OK" if self.success else "FAILED"
        if self.stopped_early:
            verdict = "STOPPED"
        parts = [
            f"flow {self.flow!r}: {verdict}",
            f"{self.steps_run} step(s) in {self.duration:.1f}s",
        ]
        if self.steps_failed:
            parts.append(f"{self.steps_failed} failed")
        if self.steps_skipped:
            parts.append(f"{self.steps_skipped} skipped")
        if self.message:
            parts.append(self.message)
        return " | ".join(parts)


class FlowRunner:
    """Runs :class:`Flow` objects against a :class:`RunContext`."""

    def __init__(self, ctx: RunContext) -> None:
        self.ctx = ctx
        self.ctx.executor = self.run_steps
        self._defaults: Dict[str, Any] = {}
        self._report = RunReport(flow="")

    # -- entry points ----------------------------------------------------
    def run(
        self,
        flow: Flow,
        *,
        variables: Optional[Dict[str, Any]] = None,
        loops: int = 1,
        loop_delay: float = 0.0,
    ) -> RunReport:
        """Run ``flow`` ``loops`` times (``loops <= 0`` means forever)."""
        report = RunReport(flow=flow.name)
        self._report = report
        self._defaults = dict(flow.defaults)

        self.ctx.variables.update(flow.vars)
        if variables:
            self.ctx.variables.update(variables)
        self.ctx.stop.reset()

        started = time.monotonic()
        iteration = 0
        try:
            while loops <= 0 or iteration < loops:
                iteration += 1
                if loops != 1:
                    log.info("=== %s: loop %d%s ===", flow.name, iteration, f"/{loops}" if loops > 0 else "")
                self.ctx.set_var("loop", iteration - 1)
                self.ctx.set_var("loop_1based", iteration)

                if not self.run_steps(flow.steps):
                    report.success = False
                if loop_delay and (loops <= 0 or iteration < loops):
                    self.ctx.sleep(loop_delay)
        except StopRequested as exc:
            report.stopped_early = True
            report.success = False
            report.message = str(exc)
            log.warning("%s", exc)
        except FlowAborted as exc:
            report.success = exc.success
            report.message = str(exc)
            log.info("Flow ended early: %s", exc)
        except KeyboardInterrupt:
            report.stopped_early = True
            report.success = False
            report.message = "interrupted (Ctrl+C)"
            log.warning("Interrupted by user")
        finally:
            report.duration = time.monotonic() - started

        log.info("%s", report.summary())
        return report

    def run_steps(self, steps: Sequence[Step]) -> bool:
        """Run a list of steps; returns False if any required step failed."""
        all_ok = True
        self.ctx.depth += 1
        try:
            for step in steps:
                if not self.run_step(step):
                    all_ok = False
        finally:
            self.ctx.depth -= 1
        return all_ok

    # -- single step -----------------------------------------------------
    def run_step(self, step: Step) -> bool:
        if not step.enabled:
            self._report.steps_skipped += 1
            log.debug("%sskip (disabled) %s", self.ctx.indent, step.label)
            return True

        self.ctx.stop.check()
        spec = step.spec
        attempts = max(1, step.retries + 1)
        last_result: Optional[ActionResult] = None
        last_error: Optional[Exception] = None

        for attempt in range(1, attempts + 1):
            self.ctx.stop.check()
            params = Params(step, self.ctx.variables, self._defaults)
            try:
                result = spec.fn(self.ctx, params)
                params.ensure_consumed()
            except (StopRequested, FlowAborted):
                raise
            except TTHeartError as exc:
                # Hard errors are retryable too -- a window can vanish mid-run.
                last_error = exc
                result = ActionResult.fail(str(exc))
            except Exception as exc:  # noqa: BLE001 - surface as a step failure
                last_error = exc
                log.exception("%sUnexpected error in %s", self.ctx.indent, step.location)
                result = ActionResult.fail(f"{type(exc).__name__}: {exc}")

            last_result = result if isinstance(result, ActionResult) else ActionResult.ok(result)
            self._report.steps_run += 1

            if last_result.success:
                if self.ctx.config.runner.step_delay:
                    self.ctx.sleep(self.ctx.config.runner.step_delay)
                return True

            if attempt < attempts:
                log.info(
                    "%sretry %d/%d for %s: %s",
                    self.ctx.indent,
                    attempt,
                    attempts - 1,
                    step.label,
                    last_result.message,
                )
                self.ctx.sleep(step.retry_delay)

        message = (last_result.message if last_result else "") or (
            str(last_error) if last_error else "step failed"
        )
        self._report.steps_failed += 1

        if step.optional:
            log.warning("%soptional step %s failed: %s", self.ctx.indent, step.label, message)
            return True

        log.error("%sstep %s failed: %s", self.ctx.indent, step.location, message)
        if last_error is not None and not isinstance(last_error, TTHeartError):
            # Programming errors should not masquerade as flow logic failures.
            raise ActionError(f"{step.location}: {message}") from last_error
        return False

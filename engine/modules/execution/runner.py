"""Closed-loop inference execution over adaptive resource scheduling."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from engine.modules.scheduling import ResourceRequest, ResourceScheduler, ResourceTier

from ._types import InferenceRequest, InferenceResult
from .registry import ExecutorRegistry


class ResilientInferenceRunner:
    """Run inference with execution feedback and rescheduling."""

    def __init__(
        self,
        *,
        scheduler: ResourceScheduler,
        registry: Optional[ExecutorRegistry] = None,
        max_attempts: int = 3,
    ) -> None:
        self.scheduler = scheduler
        self.registry = registry or ExecutorRegistry.default()
        self.max_attempts = max(1, max_attempts)

    def run(
        self,
        *,
        resource_request: ResourceRequest,
        prompt: str,
        system_prompt: str = "",
        redacted_payload: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> InferenceResult:
        attempts: List[Dict[str, Any]] = []
        last_result: Optional[InferenceResult] = None

        for attempt_index in range(1, self.max_attempts + 1):
            allocation = self.scheduler.acquire(resource_request)
            request = InferenceRequest(
                prompt=prompt,
                system_prompt=system_prompt,
                redacted_payload=redacted_payload,
                allocation=allocation.to_dict(),
                metadata={**(metadata or {}), "attempt": attempt_index},
            )

            try:
                result = self.registry.run(request)
            except Exception as exc:  # noqa: BLE001 - runner converts failures into retryable results
                result = InferenceResult(
                    text="",
                    executor="unhandled",
                    endpoint=allocation.endpoint,
                    metadata={},
                    success=False,
                    error=str(exc),
                    retryable=True,
                )

            attempts.append(
                {
                    "attempt": attempt_index,
                    "allocation": allocation.to_dict(),
                    "success": result.success,
                    "error": result.error,
                    "retryable": result.retryable,
                }
            )
            last_result = result

            if result.success:
                result.metadata = {**result.metadata, "attempts": attempts}
                return result

            self._record_failure(allocation.to_dict(), result)
            if not result.retryable:
                break

        if last_result is None:
            return InferenceResult(
                text="",
                executor="none",
                endpoint="",
                success=False,
                error="no execution attempts",
                retryable=False,
                metadata={"attempts": attempts},
            )

        last_result.metadata = {**last_result.metadata, "attempts": attempts}
        return last_result

    def _record_failure(self, allocation: Dict[str, Any], result: InferenceResult) -> None:
        monitor = getattr(self.scheduler, "monitor", None)
        record_failure = getattr(monitor, "record_failure", None)
        if not callable(record_failure):
            return

        tier = ResourceTier(str(allocation.get("tier", ResourceTier.DEVICE.value)))
        reason = result.error or str(result.metadata.get("error") or "execution_failed")
        record_failure(tier, reason)

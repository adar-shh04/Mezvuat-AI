"""Start-up negotiation with ModelMesh (contract §0, §2.4, §4; ADR-0011).

`GET /v1/contract` lists the extension fields, task types and capabilities the server accepts. Mezvuat sends only
those, because unknown fields are rejected with 400. Two fields are hard requirements and are never dropped:
`routing.data_policy` (decision D1) and `cache.namespace` (workspace isolation). Without them the server is unusable
and every call fails with `inference_contract_mismatch`.

`GET /v1/routing/calibration` says which task types are calibrated. `min_quality` is sent only for those, because an
explicit `min_quality` on an uncalibrated task type is a 422. When calibration cannot be read, nothing is calibrated.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from mezvuat.inference.errors import InferenceContractMismatchError, ModelMeshUnavailableError
from mezvuat.inference.http import error_info

log = logging.getLogger(__name__)

HARD_EXTENSIONS = ("routing.data_policy", "cache.namespace")
SUPPORTED_MAJOR = "1"


class ContractInfo(BaseModel):
    """`ContractInfo` in openapi-v1.yaml. Unknown keys are ignored: later 1.x servers may add some."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    contract_version: str
    contract_release: str
    endpoints: list[str]
    extensions: list[str]
    task_types: list[str]
    capabilities: list[str]
    routing_policies: list[str]
    feedback_metrics: list[str]

    def supports(self, path: str) -> bool:
        return path in self.extensions


class TaskCalibration(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    task_type: str
    calibrated: bool


class RoutingCalibration(BaseModel):
    """`RoutingCalibration` in openapi-v1.yaml, reduced to what the client uses."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    router_version: str | None
    generated_at: str
    task_types: list[TaskCalibration]


def check_contract(contract: ContractInfo) -> None:
    major = contract.contract_version.split(".", 1)[0]
    if major != SUPPORTED_MAJOR:
        raise InferenceContractMismatchError(
            f"ModelMesh serves contract {contract.contract_version}; this client speaks {SUPPORTED_MAJOR}.x"
        )
    missing = [path for path in HARD_EXTENSIONS if not contract.supports(path)]
    if missing:
        raise InferenceContractMismatchError(
            "ModelMesh does not support " + ", ".join(missing) + "; calls would lose data-policy or cache isolation"
        )


@dataclass
class ContractNegotiator:
    """Caches the contract (refreshed every `contract_ttl_s`) and the calibrated task types (`calibration_ttl_s`)."""

    base_url: str
    http: httpx.Client
    api_key: str | None = field(default=None, repr=False)
    contract_ttl_s: float = 600.0
    calibration_ttl_s: float = 3600.0
    timeout_s: float = 5.0
    clock: Callable[[], float] = time.monotonic
    _contract: ContractInfo | None = field(default=None, init=False)
    _contract_at: float = field(default=0.0, init=False)
    _calibrated: frozenset[str] = field(default=frozenset(), init=False)
    _calibration_at: float | None = field(default=None, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _dropped_logged: set[str] = field(default_factory=set, init=False)

    def contract(self, *, force: bool = False) -> ContractInfo:
        """The current contract. Raises `ModelMeshUnavailableError` when it cannot be read and none is cached, and
        `InferenceContractMismatchError` when the server lacks a hard requirement."""
        with self._lock:
            fresh = self._contract is not None and self.clock() - self._contract_at < self.contract_ttl_s
            if fresh and not force and self._contract is not None:
                return self._contract
            try:
                contract = self._fetch_contract()
            except ModelMeshUnavailableError:
                if self._contract is not None and not force:
                    log.warning("ModelMesh contract refresh failed; keeping the cached contract")
                    return self._contract
                raise
            self._contract, self._contract_at = contract, self.clock()
            return contract

    def _fetch_contract(self) -> ContractInfo:
        try:
            response = self.http.get(f"{self.base_url}/v1/contract", timeout=self.timeout_s)
        except httpx.TransportError as exc:
            raise ModelMeshUnavailableError(
                f"ModelMesh contract unreachable: {type(exc).__name__}", reason="modelmesh_unreachable"
            ) from exc
        if response.status_code != 200:
            info = error_info(response)
            raise ModelMeshUnavailableError(
                f"ModelMesh contract returned {info.summary}", reason=f"modelmesh_contract_{info.status}"
            )
        try:
            contract = ContractInfo.model_validate_json(response.content)
        except ValidationError as exc:
            raise InferenceContractMismatchError(
                "ModelMesh returned a contract document this client cannot read"
            ) from exc
        check_contract(contract)
        log.info(
            "ModelMesh contract negotiated",
            extra={"contract_version": contract.contract_version, "extensions": len(contract.extensions)},
        )
        return contract

    def calibrated(self) -> frozenset[str]:
        """Task types with `calibrated: true`. Any failure means none, which only ever omits `min_quality`."""
        with self._lock:
            if self._calibration_at is not None and self.clock() - self._calibration_at < self.calibration_ttl_s:
                return self._calibrated
            self._calibration_at = self.clock()
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            try:
                response = self.http.get(
                    f"{self.base_url}/v1/routing/calibration", headers=headers, timeout=self.timeout_s
                )
                if response.status_code != 200:
                    raise ValueError(error_info(response).summary)
                report = RoutingCalibration.model_validate_json(response.content)
            except (httpx.TransportError, ValueError, ValidationError) as exc:
                log.warning(
                    "ModelMesh calibration unavailable; min_quality will not be sent", extra={"error": str(exc)[:200]}
                )
                # Try again in a minute rather than an hour: ModelMesh may just have been starting.
                self._calibration_at = self.clock() - self.calibration_ttl_s + 60.0
                self._calibrated = frozenset()
                return self._calibrated
            self._calibrated = frozenset(t.task_type for t in report.task_types if t.calibrated)
            return self._calibrated

    def note_dropped(self, path: str) -> None:
        """Logs once per process that an extension is not sent because the server does not list it."""
        if path not in self._dropped_logged:
            self._dropped_logged.add(path)
            log.warning("extension_unsupported:%s", path)

    def negotiate(self) -> ContractInfo:
        """Start-up: read both documents now. Raises the same errors as `contract()`."""
        contract = self.contract(force=True)
        self.calibrated()
        return contract

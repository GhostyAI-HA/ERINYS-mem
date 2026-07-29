"""Typed contracts for ERINYS Current Valid Claim shadow retrieval."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ._compat import StrEnum

CONTRACT_SCHEMA_VERSION = 1
MIN_ATTRIBUTION_CONFIDENCE = 0.7


class ContractModel(BaseModel):
    """Strict base model for the v0 contract."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class NodeType(StrEnum):
    CLAIM = "CLAIM"
    EVIDENCE = "EVIDENCE"
    SOURCE = "SOURCE"
    DECISION = "DECISION"
    ACTION = "ACTION"
    OUTCOME = "OUTCOME"
    RECEIPT = "RECEIPT"
    POLICY = "POLICY"


class EdgeType(StrEnum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    SUPERSEDES = "supersedes"
    NARROWS = "narrows"
    DERIVED_FROM = "derived_from"
    APPLICABLE_TO = "applicable_to"
    USED_IN = "used_in"
    VERIFIED_BY = "verified_by"
    RESULTED_IN = "resulted_in"


class ClaimStatus(StrEnum):
    ACTIVE = "ACTIVE"
    SUPERSEDED = "SUPERSEDED"
    CONTESTED = "CONTESTED"
    REFUTED = "REFUTED"
    RETRACTED = "RETRACTED"
    ARCHIVED = "ARCHIVED"


class VerificationState(StrEnum):
    UNVERIFIED = "UNVERIFIED"
    SYSTEM_VERIFIED = "SYSTEM_VERIFIED"
    HUMAN_APPROVED = "HUMAN_APPROVED"
    REJECTED = "REJECTED"


class EdgeOrigin(StrEnum):
    DETERMINISTIC = "DETERMINISTIC"
    VERIFIED_DERIVED = "VERIFIED_DERIVED"
    CANDIDATE = "CANDIDATE"


class EdgeStatus(StrEnum):
    CANDIDATE = "CANDIDATE"
    VERIFIED = "VERIFIED"
    REJECTED = "REJECTED"
    RETRACTED = "RETRACTED"


class QueryMode(StrEnum):
    LOOKUP = "lookup"
    CURRENT_VALID_CLAIM = "current_valid_claim"
    PATH = "path"
    TEMPORAL = "temporal"
    COUNTEREVIDENCE = "counterevidence"
    AUDIT = "audit"


class RiskClass(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class OutcomeStatus(StrEnum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    MIXED = "MIXED"
    UNKNOWN = "UNKNOWN"


class AttributionStatus(StrEnum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    UNRESOLVED = "UNRESOLVED"


class ReinforcementTarget(StrEnum):
    RETRIEVAL_PRIORITY = "RETRIEVAL_PRIORITY"
    RELIABILITY = "RELIABILITY"
    APPLICABILITY = "APPLICABILITY"
    REUSE_CANDIDACY = "REUSE_CANDIDACY"


class ClaimScope(ContractModel):
    corpus: str | None = None
    project: str | None = None
    tenant: str | None = None
    environment: str | None = None
    jurisdiction: str | None = None
    product_version: str | None = None
    source_version: str | None = None

    @model_validator(mode="after")
    def require_dimension(self) -> ClaimScope:
        if all(value is None for value in self.model_dump().values()):
            raise ValueError("scope requires at least one dimension")
        return self


class ValidTime(ContractModel):
    from_: datetime = Field(alias="from")
    to: datetime | None = None

    @model_validator(mode="after")
    def validate_interval(self) -> ValidTime:
        if self.to is not None and self.to <= self.from_:
            raise ValueError("valid_time.to must be after valid_time.from")
        return self


class SourceProvenance(ContractModel):
    source_id: str = Field(min_length=1)
    source_type: str = Field(min_length=1)
    locator: str | None = None
    acquired_at: datetime | None = None


class TransformationLineage(ContractModel):
    transformation_id: str = Field(min_length=1)
    operation: str = Field(min_length=1)
    parent_node_ids: list[str] = Field(default_factory=list)
    model: str | None = None
    implementation_version: str | None = None


class ExecutionProvenance(ContractModel):
    decision_id: str | None = None
    receipt_ids: list[str] = Field(default_factory=list)
    outcome_ids: list[str] = Field(default_factory=list)


class ProvenanceBundle(ContractModel):
    sources: list[SourceProvenance] = Field(default_factory=list)
    transformations: list[TransformationLineage] = Field(default_factory=list)
    executions: list[ExecutionProvenance] = Field(default_factory=list)


class ClaimNode(ContractModel):
    schema_version: Literal[1] = CONTRACT_SCHEMA_VERSION
    node_type: Literal["CLAIM"] = "CLAIM"
    claim_id: str = Field(min_length=1)
    proposition: str = Field(min_length=1)
    scope: ClaimScope
    valid_time: ValidTime
    recorded_at: datetime
    status: ClaimStatus
    confidence: float = Field(ge=0.0, le=1.0)
    verification_state: VerificationState = VerificationState.UNVERIFIED
    risk_class: RiskClass = RiskClass.LOW
    legal_retention: bool = False
    historical_value: bool = False
    provenance: ProvenanceBundle

    @model_validator(mode="after")
    def require_source_provenance(self) -> ClaimNode:
        if not self.provenance.sources:
            raise ValueError("claim requires source provenance")
        return self

    @model_validator(mode="after")
    def require_historical_closure(self) -> ClaimNode:
        historical = {ClaimStatus.SUPERSEDED, ClaimStatus.REFUTED, ClaimStatus.RETRACTED}
        if self.status in historical and self.valid_time.to is None:
            raise ValueError("historical claim requires valid_time.to")
        return self

    def retention_protected(self) -> bool:
        protected_risks = {RiskClass.HIGH, RiskClass.CRITICAL}
        return self.legal_retention or self.historical_value or self.risk_class in protected_risks


class MemoryNode(ContractModel):
    schema_version: Literal[1] = CONTRACT_SCHEMA_VERSION
    node_id: str = Field(min_length=1)
    node_type: NodeType
    content: str = Field(min_length=1)
    scope: ClaimScope
    recorded_at: datetime
    provenance: ProvenanceBundle


class MemoryEdge(ContractModel):
    schema_version: Literal[1] = CONTRACT_SCHEMA_VERSION
    edge_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    relation: EdgeType
    origin: EdgeOrigin
    status: EdgeStatus
    confidence: float = Field(ge=0.0, le=1.0)
    valid_time: ValidTime | None = None
    verification_id: str | None = None

    @model_validator(mode="after")
    def validate_trust_state(self) -> MemoryEdge:
        if self.source_id == self.target_id:
            raise ValueError("self-referential edge is not allowed")
        if self.status is EdgeStatus.CANDIDATE and self.origin is not EdgeOrigin.CANDIDATE:
            raise ValueError("candidate status requires candidate origin")
        if self.status is EdgeStatus.VERIFIED and self.origin is EdgeOrigin.CANDIDATE:
            raise ValueError("candidate origin cannot be verified directly")
        return self


class EvidencePath(ContractModel):
    """Evidence retrieval path; never a representation of hidden reasoning."""

    path_id: str = Field(min_length=1)
    node_ids: list[str] = Field(min_length=2)
    edge_ids: list[str] = Field(min_length=1)
    relations: list[EdgeType] = Field(min_length=1)
    score: float = Field(ge=0.0)
    selected_reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_shape(self) -> EvidencePath:
        if len(self.edge_ids) != len(self.node_ids) - 1:
            raise ValueError("edge count must equal node count minus one")
        if len(self.relations) != len(self.edge_ids):
            raise ValueError("relation count must equal edge count")
        return self


class ClaimSearchRequest(ContractModel):
    query: str = Field(min_length=1)
    as_of: datetime
    knowledge_as_of: datetime
    scope: ClaimScope
    mode: QueryMode = QueryMode.CURRENT_VALID_CLAIM
    max_hops: int = Field(default=2, ge=0, le=2)


class RetrievalTrace(ContractModel):
    vector_used: bool
    graph_used: bool
    max_hops: int = Field(ge=0, le=2)
    strategy: list[str] = Field(default_factory=list)
    baseline_node_ids: list[str] = Field(default_factory=list)
    visited_node_ids: list[str] = Field(default_factory=list)
    returned_node_ids: list[str] = Field(default_factory=list)
    discarded_node_ids: list[str] = Field(default_factory=list)
    selected_path_ids: list[str] = Field(default_factory=list)
    discarded_path_ids: list[str] = Field(default_factory=list)
    fallback_reason: str | None = None


class CurrentValidClaimResult(ContractModel):
    schema_version: Literal[1] = CONTRACT_SCHEMA_VERSION
    request: ClaimSearchRequest
    active_claims: list[ClaimNode] = Field(default_factory=list)
    supporting_paths: list[EvidencePath] = Field(default_factory=list)
    conflicting_paths: list[EvidencePath] = Field(default_factory=list)
    historical_claims: list[ClaimNode] = Field(default_factory=list)
    uncertainty: list[str] = Field(default_factory=list)
    retrieval_trace: RetrievalTrace


class OutcomeAttributionReceipt(ContractModel):
    memory_id: str = Field(min_length=1)
    usage: str = Field(min_length=1)
    decision_id: str = Field(min_length=1)
    outcome_id: str = Field(min_length=1)
    outcome_status: OutcomeStatus
    attribution_status: AttributionStatus
    attribution_confidence: float = Field(ge=0.0, le=1.0)
    verified_by: list[str] = Field(default_factory=list)
    scope: ClaimScope
    reinforcement_targets: list[ReinforcementTarget] = Field(default_factory=list)

    def reinforcement_eligible(self) -> bool:
        positive = self.outcome_status is OutcomeStatus.POSITIVE
        accepted = self.attribution_status is AttributionStatus.ACCEPTED
        confident = self.attribution_confidence >= MIN_ATTRIBUTION_CONFIDENCE
        evidenced = bool(self.verified_by) and bool(self.reinforcement_targets)
        return positive and accepted and confident and evidenced

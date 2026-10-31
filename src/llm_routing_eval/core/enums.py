"""Stable, domain-independent values shared by benchmark components."""

from enum import StrEnum


class StrategyId(StrEnum):
    ALWAYS_LARGE = "always_large"
    RULE_BASED = "rule_based"
    KNN = "knn"
    CASCADE = "cascade"


class RoutingAction(StrEnum):
    INVOKE_MODEL = "INVOKE_MODEL"
    FINISH = "FINISH"


class CallStatus(StrEnum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    RATE_LIMITED = "RATE_LIMITED"


class ExecutionStatus(StrEnum):
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"


class EvaluationStatus(StrEnum):
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    NOT_EVALUATED = "NOT_EVALUATED"
    FAILED = "FAILED"


class VerificationStatus(StrEnum):
    UNREVIEWED = "UNREVIEWED"
    REVIEW_PENDING = "REVIEW_PENDING"
    EXPERT_VERIFIED = "EXPERT_VERIFIED"


class MetricOrigin(StrEnum):
    SIMULATED = "SIMULATED"
    OBSERVED = "OBSERVED"


class CostOrigin(StrEnum):
    BILLED = "BILLED"
    ESTIMATED = "ESTIMATED"
    SIMULATED = "SIMULATED"
    UNKNOWN = "UNKNOWN"


class ErrorCategory(StrEnum):
    VALIDATION = "VALIDATION"
    PRIVACY = "PRIVACY"
    PROVIDER = "PROVIDER"
    PROVIDER_TIMEOUT = "PROVIDER_TIMEOUT"
    RATE_LIMIT = "RATE_LIMIT"
    ROUTING = "ROUTING"
    EVALUATION = "EVALUATION"
    PERSISTENCE = "PERSISTENCE"
    UNKNOWN = "UNKNOWN"


class TraceStepKind(StrEnum):
    ROUTING_DECISION = "ROUTING_DECISION"
    PROVIDER_ATTEMPT = "PROVIDER_ATTEMPT"
    PRIVACY_CHECK = "PRIVACY_CHECK"
    EVALUATION = "EVALUATION"
    ERROR = "ERROR"


class SanitizationStatus(StrEnum):
    ALLOWED = "ALLOWED"
    BLOCKED = "BLOCKED"

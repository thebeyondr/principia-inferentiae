"""In-memory student state. Persistence lives in the CLI."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from principia_inferentiae.practice.dag import GradingType, Status


class Choice(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    label: str


class AttemptRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    attempt_id: str
    seq: int
    concept_id: str
    seed: int
    difficulty: float
    prompt: str
    canonical_answer: str
    facts: dict[str, object]
    grading_type: GradingType
    choices: tuple[Choice, ...] | None = None
    unit: str | None = None
    tolerance: float = 0.02
    order_threshold: float = 0.8
    label: str
    domain: str
    served_at: datetime
    answered_at: datetime | None = None
    response: str | None = None
    correct: bool | None = None
    cold: bool = False
    seconds: float | None = None
    instance_key: str
    origin: str = "practice"


class ConceptProgress(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    concept_id: str
    status: Status = Status.UNSEEN
    mastery_score: float = 0.0
    weakness: float = 0.0
    repetitions: int = 0
    easiness: float = 2.5
    interval_days: int = 0
    next_due: datetime | None = None
    last_seen: datetime | None = None
    streak: int = 0


class ProbeRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    concept_id: str
    correct: bool
    slow: bool


class StudentState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    progress: dict[str, ConceptProgress] = Field(default_factory=dict)
    attempts: tuple[AttemptRecord, ...] = ()
    probes: tuple[ProbeRecord, ...] = ()
    last_concept_id: str | None = None
    last_subdomain: str | None = None

    def answered(self) -> tuple[AttemptRecord, ...]:
        return tuple(attempt for attempt in self.attempts if attempt.correct is not None)

    def progress_for(self, concept_id: str) -> ConceptProgress | None:
        return self.progress.get(concept_id)

    def with_progress(self, progress: ConceptProgress) -> StudentState:
        updated = dict(self.progress)
        updated[progress.concept_id] = progress
        return self.model_copy(update={"progress": updated})

    def with_attempt(self, attempt: AttemptRecord) -> StudentState:
        return self.model_copy(update={"attempts": (*self.attempts, attempt)})

    def replace_attempt(self, attempt: AttemptRecord) -> StudentState:
        replaced = tuple(
            attempt if item.attempt_id == attempt.attempt_id else item for item in self.attempts
        )
        return self.model_copy(update={"attempts": replaced})

    def with_probe(self, probe: ProbeRecord) -> StudentState:
        return self.model_copy(update={"probes": (*self.probes, probe)})

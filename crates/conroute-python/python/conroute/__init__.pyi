from typing import Optional

class Candidate:
    content: str
    model: Optional[str]
    confidence: Optional[float]

    def __init__(
        self,
        content: str,
        model: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> None: ...

class ConsensusResult:
    content: str
    strategy: str
    agreement_score: float
    reasoning: Optional[str]
    dissent_count: int

    def to_dict(self) -> dict: ...

def consensus(
    candidates: list[Candidate],
    strategy: str = "majority_vote",
) -> ConsensusResult: ...

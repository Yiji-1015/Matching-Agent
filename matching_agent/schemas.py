from typing import Literal, Optional

from pydantic import BaseModel
from typing_extensions import TypedDict


class OrchestratorResponse(BaseModel):
    hypernym: Optional[str]
    matching_target: Optional[str]
    reasoning: str
    next_agent: Literal[
        "Web_Search_Module",
        "Message_Analyzer",
        "Query_Reformer",
        "Selector",
        "Evaluator",
        "__end__",
    ]
    next_action: str


class MessageAnalyzerResponse(BaseModel):
    hypernym: Optional[str]
    matching_target: Optional[str]
    reasoning: str
    next_agent: Literal["Orchestrator", "Web_Search_Module"]
    next_action: str


class QueryReformerResponse(BaseModel):
    result: str
    reasoning: str
    next_agent: Literal["Selector", "Web_Search_Module"]
    next_action: str


class SelectorScore(TypedDict):
    candidate: str
    coherence_with_user: str
    score: float


class SubSelectorResponse(BaseModel):
    opinion: str


class RoleSelectorResponse(BaseModel):
    user_role: str
    candidates: list[SelectorScore]


class SelectorResponse(BaseModel):
    matched_message: str
    matched_username: str
    reasoning: str
    next_agent: Literal["Evaluator", "Orchestrator"]
    next_action: str
    certainty: int


class EvaluatorResponse(BaseModel):
    matched_username: str
    matched_message: str
    fail_or_not: str
    reasoning: str
    next_agent: Literal["Orchestrator", "__end__"]
    next_action: str
    success_or_fail: str
    detail: str
    failed_query: str
    matched_candidate: str
    certainty: int

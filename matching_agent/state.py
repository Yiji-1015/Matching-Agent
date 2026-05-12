import operator
from typing import Annotated, Optional

from langchain_core.messages import BaseMessage, HumanMessage
from typing_extensions import TypedDict


class FailureLog(TypedDict):
    detail: str
    failed_query: str
    failed_matched_mesasge: str
    failed_matched_candidate: str
    certainty: int


class AgentState(TypedDict):
    input_message: str
    username: str
    count: int
    evaluation_count: int
    hypernym: str
    matching_target: str
    matched_message: Optional[str]
    matched_username: Optional[str]
    certainty: Optional[int]
    fail_or_not: str
    messages: str
    last_agent: str
    next_agent: str
    analyzed_message: Optional[str]
    search_info: Optional[str]
    reformed_query: Optional[str]
    candidates: Optional[list[str]]
    selectors_history: list[str]
    failure_log: list[FailureLog]
    history: Annotated[list[BaseMessage], operator.add]


def get_initial_state(input_message: str, username: str = "Guest") -> AgentState:
    return {
        "input_message": input_message,
        "username": username,
        "count": 0,
        "evaluation_count": 1,
        "hypernym": "",
        "matching_target": "",
        "matched_message": None,
        "matched_username": None,
        "certainty": None,
        "fail_or_not": "",
        "messages": input_message,
        "history": [HumanMessage(content=input_message)],
        "next_agent": "Orchestrator",
        "last_agent": "User",
        "analyzed_message": None,
        "search_info": None,
        "reformed_query": None,
        "selectors_history": ["."],
        "candidates": None,
        "failure_log": [],
    }

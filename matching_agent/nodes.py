import functools

from langchain.tools import tool
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from .config import DEFAULT_LLM_MODEL
from .prompts import load_prompt
from .retriever import dedupe, get_candidates
from .schemas import (
    EvaluatorResponse,
    MessageAnalyzerResponse,
    OrchestratorResponse,
    QueryReformerResponse,
    RoleSelectorResponse,
    SelectorResponse,
    SubSelectorResponse,
)


@tool
def web_search_tool(query: str):
    """Run external web search and return compact content snippets."""
    web_search = TavilySearchResults(max_results=5)
    search_results = web_search.invoke(query)
    return [item.get("content") for item in search_results]


class MatchingAgentNodes:
    def __init__(self, retriever, *, llm_model: str = DEFAULT_LLM_MODEL):
        self.retriever = retriever
        self.gpt = ChatOpenAI(model_name=llm_model, temperature=0)

        self.orchestrator_prompt = load_prompt("Orchestrator.txt")
        self.message_analyzer_prompt = load_prompt("Message_Analyzer.txt")
        self.query_reformer_prompt = load_prompt("Query_Reformer.txt")
        self.persona_prompt = load_prompt("Selector_PersonaMatch.txt")
        self.role_prompt = load_prompt("Selector_RoleMatch.txt")
        self.type_prompt = load_prompt("Selector_TypeMatch.txt")
        self.selector_prompt = load_prompt("Selector.txt")
        self.evaluator_prompt = load_prompt("Evaluator.txt")
        self.web_search_prompt = load_prompt("Web_Search_Module.txt")

        self.web_search_module = create_react_agent(
            self.gpt,
            tools=[web_search_tool],
            state_modifier=self.web_search_prompt,
        )
        self.web_search_node = functools.partial(
            self.web_search,
            agent=self.web_search_module,
        )

    def orchestrator(self, state: dict):
        prompt = ChatPromptTemplate.from_messages(
            [("system", self.orchestrator_prompt)]
        ).partial(
            recent_history=state["history"][-5:],
            analyzed_message=state["analyzed_message"],
            search_info=state["search_info"],
        )
        response = (prompt | self.gpt.with_structured_output(OrchestratorResponse)).invoke(state)

        state.update(
            {
                "hypernym": response.hypernym or "",
                "matching_target": response.matching_target or "",
                "next_agent": response.next_agent,
                "messages": response.reasoning + "\n" + response.next_action,
                "history": [
                    AIMessage(
                        content=response.reasoning + "\n" + response.next_action,
                        name="Orchestrator",
                    )
                ],
                "last_agent": "Orchestrator",
            }
        )

        if (
            state["analyzed_message"] is None
            and state["candidates"] is None
            and state["next_agent"] == "__end__"
        ):
            state["next_agent"] = "Message_Analyzer"

        return state

    def message_analyzer(self, state: dict):
        if state["count"] > 7:
            message = "현재 병목이 발생하는 것으로 보이니, Query_Reformer로 넘어가겠습니다."
            state.update(
                {
                    "next_agent": "Orchestrator",
                    "messages": message,
                    "history": [AIMessage(content=message, name="Message_Analyzer")],
                    "last_agent": "Message_Analyzer",
                }
            )
            return state

        prompt = ChatPromptTemplate.from_messages(
            [("system", self.message_analyzer_prompt)]
        ).partial(recent_history=state["history"][-5:])
        response = (prompt | self.gpt.with_structured_output(MessageAnalyzerResponse)).invoke(state)

        state.update(
            {
                "hypernym": response.hypernym or "",
                "matching_target": response.matching_target or "",
                "next_agent": response.next_agent,
                "messages": response.reasoning + "\n" + response.next_action,
                "history": [
                    AIMessage(
                        content=response.reasoning + "\n" + response.next_action,
                        name="Message_Analyzer",
                    )
                ],
                "last_agent": "Message_Analyzer",
                "analyzed_message": response.reasoning + "\n" + response.next_action,
            }
        )
        return state

    def web_search(self, state: dict, agent):
        response = agent.invoke(state)
        state.update(
            {
                "next_agent": state["last_agent"],
                "messages": response["messages"][-1].content,
                "history": [
                    AIMessage(
                        content=response["messages"][-1].content,
                        name="Web_Search_Module",
                    )
                ],
                "last_agent": "Web_Search_Module",
                "search_info": response["messages"][-1].content,
            }
        )
        return state

    def query_reformer(self, state: dict):
        prompt = ChatPromptTemplate.from_messages([("system", self.query_reformer_prompt)])
        response = (prompt | self.gpt.with_structured_output(QueryReformerResponse)).invoke(state)
        reformed_queries = [part.strip() for part in response.result.split("/") if part.strip()]

        candidates = get_candidates(state["input_message"], state["username"], self.retriever)
        for query in reformed_queries:
            candidates.extend(get_candidates(query, state["username"], self.retriever))

        state.update(
            {
                "next_agent": response.next_agent,
                "messages": (
                    f"Reformed Queries - {response.result}, "
                    f"Reasoning - {response.reasoning}\n{response.next_action}"
                ),
                "reformed_query": response.result,
                "history": [
                    AIMessage(
                        content=(
                            f"Reformed Queries - {response.result}, "
                            f"Reasoning - {response.reasoning}\n{response.next_action}"
                        ),
                        name="Query_Reformer",
                    )
                ],
                "last_agent": "Query_Reformer",
                "candidates": dedupe(candidates),
            }
        )
        return state

    def persona_match(self, state: dict):
        prompt = ChatPromptTemplate.from_messages([("system", self.persona_prompt)])
        response = (prompt | self.gpt.with_structured_output(SubSelectorResponse)).invoke(state)
        state["selectors_history"] = state["selectors_history"] + [
            f"This is from PersonaMatch\n: {response.opinion}"
        ]
        return state

    def role_match(self, state: dict):
        prompt = ChatPromptTemplate.from_messages([("system", self.role_prompt)])
        response = (prompt | self.gpt.with_structured_output(RoleSelectorResponse)).invoke(state)
        state["selectors_history"] = state["selectors_history"] + [
            f"This is from RoleMatch: {response.user_role}\n{response.candidates}"
        ]
        return state

    def type_match(self, state: dict):
        prompt = ChatPromptTemplate.from_messages([("system", self.type_prompt)])
        response = (prompt | self.gpt.with_structured_output(SubSelectorResponse)).invoke(state)
        state["selectors_history"] = state["selectors_history"] + [
            f"This is from TypeMatch\n{response.opinion}"
        ]
        return state

    def selector(self, state: dict):
        prompt = ChatPromptTemplate.from_messages([("system", self.selector_prompt)])
        response = (prompt | self.gpt.with_structured_output(SelectorResponse)).invoke(state)
        state.update(
            {
                "next_agent": response.next_agent,
                "messages": response.reasoning + " " + response.next_action,
                "history": [
                    AIMessage(
                        content=response.reasoning + " " + response.next_action,
                        name="Selector",
                    )
                ],
                "last_agent": "Selector",
                "matched_message": response.matched_message,
                "matched_username": response.matched_username,
                "certainty": response.certainty,
            }
        )
        return state

    def selection(self, state: dict):
        state["selectors_history"] = []
        state = self.type_match(state)
        state = self.role_match(state)
        state = self.persona_match(state)
        return self.selector(state)

    def evaluator(self, state: dict):
        if not state["certainty"]:
            message = "Query Reform 및 Select 과정을 무조건 거쳐야 합니다."
            state.update(
                {
                    "next_agent": "Orchestrator",
                    "messages": message,
                    "history": [AIMessage(content=message, name="Evaluator")],
                    "last_agent": "Evaluator",
                }
            )
            return state

        candidates = (
            "일단은 Selector가 선택한 후보만 평가해보세요."
            if state["evaluation_count"] == 2
            else state["candidates"]
        )
        prompt = ChatPromptTemplate.from_messages(
            [("system", self.evaluator_prompt)]
        ).partial(
            chat_history=state["history"][-10:],
            candidates=candidates,
        )
        response = (prompt | self.gpt.with_structured_output(EvaluatorResponse)).invoke(state)

        updated_state = {
            "fail_or_not": response.fail_or_not,
            "next_agent": response.next_agent,
            "messages": response.reasoning + "\n" + response.next_action,
            "history": [
                AIMessage(
                    content=response.reasoning + " " + response.next_action,
                    name="Evaluator",
                )
            ],
            "evaluation_count": state["evaluation_count"] + 1,
            "last_agent": "Evaluator",
            "matched_username": response.matched_username,
            "matched_message": response.matched_message,
        }

        if response.fail_or_not == "fail":
            fail_already_logged = any(
                log.get("failed_matched_candidate") == response.matched_candidate
                and log.get("failed_query") == response.failed_query
                for log in state.get("failure_log", [])
            )
            if not fail_already_logged:
                state["failure_log"] = state["failure_log"] + [
                    {
                        "detail": response.reasoning + "\n" + response.detail,
                        "failed_query": response.failed_query,
                        "failed_matched_mesasge": response.matched_message,
                        "failed_matched_candidate": response.matched_candidate,
                        "certainty": response.certainty,
                    }
                ]

        if updated_state["evaluation_count"] == 3:
            updated_state["next_agent"] = "__end__"

        state.update(updated_state)
        return state

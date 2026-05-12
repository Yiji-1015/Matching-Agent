from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .nodes import MatchingAgentNodes
from .retriever import load_retriever
from .state import AgentState


def build_matching_graph(*, retriever=None, checkpointer=None):
    retriever = retriever or load_retriever()
    nodes = MatchingAgentNodes(retriever)

    graph = StateGraph(AgentState)
    graph.add_node("Orchestrator", nodes.orchestrator)
    graph.add_node("Web_Search_Module", nodes.web_search_node)
    graph.add_node("Message_Analyzer", nodes.message_analyzer)
    graph.add_node("Query_Reformer", nodes.query_reformer)
    graph.add_node("Selector", nodes.selection)
    graph.add_node("Evaluator", nodes.evaluator)

    graph.add_edge(START, "Orchestrator")
    graph.add_conditional_edges(
        "Orchestrator",
        lambda state: state["next_agent"],
        {
            "Web_Search_Module": "Web_Search_Module",
            "Message_Analyzer": "Message_Analyzer",
            "Query_Reformer": "Query_Reformer",
            "Selector": "Selector",
            "Evaluator": "Evaluator",
            "__end__": END,
        },
    )
    graph.add_conditional_edges(
        "Web_Search_Module",
        lambda state: state["next_agent"],
        {
            "Orchestrator": "Orchestrator",
            "Message_Analyzer": "Message_Analyzer",
        },
    )
    graph.add_conditional_edges(
        "Message_Analyzer",
        lambda state: state["next_agent"],
        {
            "Orchestrator": "Orchestrator",
            "Web_Search_Module": "Web_Search_Module",
        },
    )
    graph.add_conditional_edges(
        "Query_Reformer",
        lambda state: state["next_agent"],
        {"Selector": "Selector"},
    )
    graph.add_conditional_edges(
        "Selector",
        lambda state: state["next_agent"],
        {
            "Orchestrator": "Orchestrator",
            "Evaluator": "Evaluator",
        },
    )
    graph.add_conditional_edges(
        "Evaluator",
        lambda state: state["next_agent"],
        {
            "__end__": END,
            "Orchestrator": "Orchestrator",
        },
    )

    return graph.compile(checkpointer=checkpointer or MemorySaver())

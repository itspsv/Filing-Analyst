"""The same agent as loop.py, as a LangGraph state graph with a SQLite checkpointer.

    agent --(tool calls)--> tools --(within limits)--> agent
      |                       |
      +--(done or limit)--> compose --> END

Prompts, tools, limits and citation checks are shared with loop.py; only the control flow
moves into the graph. The state holds plain JSON-like values, so every step can be saved.

    uv run python -m filing_analyst.agent.graph "What was NVIDIA's gross margin in FY2026?"
"""

import argparse
import uuid
from collections.abc import Callable
from typing import Any, Literal, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from filing_analyst.agent.compose import Evidence, compose_answer
from filing_analyst.agent.loop import (
    COMPOSE_RESERVE,
    INVALID_TOOL_CALL,
    MAX_STEPS,
    STEP_MAX_TOKENS,
    SYSTEM_PROMPT,
    TOKEN_BUDGET,
    AgentClient,
    AgentResult,
    Step,
    describe,
    print_step,
)
from filing_analyst.agent.tools import TOOL_SPECS, ToolRunner
from filing_analyst.answer import Answer, load_default_search
from filing_analyst.config import Settings
from filing_analyst.llm import CallStats, DailyBudgetExceeded, LLMClient, LLMError
from filing_analyst.parse.chunk import Chunk
from filing_analyst.tools.compute_ratio import RatioResult
from filing_analyst.tools.query_financials import QueryResult

Next = Literal["tools", "agent", "compose"]


class AgentState(TypedDict, total=False):
    question: str
    messages: list[dict[str, Any]]
    pending: list[dict[str, Any]]  # tool calls the last agent step asked for
    next: Next
    step: int
    llm_calls: int
    tokens_in: int
    tokens_out: int
    chunks: dict[str, dict]  # evidence as plain dicts, so the checkpointer can save it
    queries: dict[str, dict]
    ratios: list[dict]
    trace: list[dict]
    stop_reason: str
    answer: dict
    untraced: list[str]


def build_graph(
    client: AgentClient,
    runner: ToolRunner,
    max_steps: int = MAX_STEPS,
    token_budget: int = TOKEN_BUDGET,
    checkpointer: BaseCheckpointSaver | None = None,
    on_step: Callable[[Step], None] | None = None,
) -> CompiledStateGraph:
    def record(state: AgentState, step: Step) -> list[dict]:
        if on_step:
            on_step(step)
        return [*state["trace"], step.model_dump()]

    def agent(state: AgentState) -> AgentState:
        if state["tokens_in"] + state["tokens_out"] >= token_budget - COMPOSE_RESERVE:
            return {"next": "compose", "stop_reason": f"token budget ({token_budget:,})"}
        step = state["step"] + 1
        try:
            message, stats = client.chat_with_tools(
                state["messages"], TOOL_SPECS, "agent_step", STEP_MAX_TOKENS
            )
        except DailyBudgetExceeded:
            raise
        except LLMError as err:
            if INVALID_TOOL_CALL not in str(err).lower():
                return {"step": step, "next": "compose", "stop_reason": f"LLM error: {err}"[:200]}
            note = f"Your last tool call was rejected: {str(err)[:300]}. Try again with "
            failed = Step(number=step, tool="(invalid tool call)", arguments="",
                          result=f"ERROR {str(err)[:150]}", error=True)  # fmt: skip
            messages = [*state["messages"], {"role": "user", "content": note + "valid arguments."}]
            go_on: Next = "agent" if step < max_steps else "compose"
            return {"step": step, "messages": messages, "trace": record(state, failed),
                    "next": go_on}  # fmt: skip
        spent = {"step": step, "llm_calls": state["llm_calls"] + 1,
                 "tokens_in": state["tokens_in"] + stats.tokens_in,
                 "tokens_out": state["tokens_out"] + stats.tokens_out}  # fmt: skip
        if not message["tool_calls"]:
            return {**spent, "next": "compose", "stop_reason": "model finished"}
        messages = [*state["messages"], message]
        return {**spent, "messages": messages, "pending": message["tool_calls"], "next": "tools"}

    def tools(state: AgentState) -> AgentState:
        messages, trace = list(state["messages"]), list(state["trace"])
        chunks, queries, ratios = (
            dict(state["chunks"]),
            dict(state["queries"]),
            list(state["ratios"]),
        )
        for call in state["pending"]:
            name, arguments = call["function"]["name"], call["function"]["arguments"]
            outcome = runner.run(name, arguments)
            chunks |= {c.chunk_id: c.model_dump() for c in outcome.chunks}
            if outcome.query:
                queries[outcome.query.query_id] = outcome.query.model_dump()
            if outcome.ratio:
                ratios.append(outcome.ratio.model_dump())
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": outcome.text})
            step = Step(number=state["step"], tool=name, arguments=arguments,
                        result=describe(outcome), error=outcome.error is not None)  # fmt: skip
            trace = record({**state, "trace": trace}, step)
        go_on: Next = "agent" if state["step"] < max_steps else "compose"
        return {"messages": messages, "trace": trace, "chunks": chunks, "queries": queries,
                "ratios": ratios, "pending": [], "next": go_on}  # fmt: skip

    def compose(state: AgentState) -> AgentState:
        evidence = Evidence(
            chunks={k: Chunk(**v) for k, v in state["chunks"].items()},
            queries={k: QueryResult(**v) for k, v in state["queries"].items()},
            ratios=[RatioResult(**r) for r in state["ratios"]],
        )
        spent = CallStats(tokens_in=state["tokens_in"], tokens_out=state["tokens_out"])
        answer, untraced = compose_answer(
            state["question"], evidence, client, spent, state["llm_calls"]
        )
        return {"answer": answer.model_dump(), "untraced": untraced}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", tools)
    graph.add_node("compose", compose)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", lambda s: s["next"], ["tools", "agent", "compose"])
    graph.add_conditional_edges("tools", lambda s: s["next"], ["agent", "compose"])
    graph.add_edge("compose", END)
    return graph.compile(checkpointer=checkpointer)


def run_graph(
    question: str,
    client: AgentClient,
    runner: ToolRunner,
    max_steps: int = MAX_STEPS,
    token_budget: int = TOKEN_BUDGET,
    on_step: Callable[[Step], None] | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    thread_id: str | None = None,
) -> AgentResult:
    graph = build_graph(client, runner, max_steps, token_budget, checkpointer, on_step)
    start: AgentState = {
        "question": question,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": question}],
        "pending": [], "step": 0, "llm_calls": 0, "tokens_in": 0, "tokens_out": 0,
        "chunks": {}, "queries": {}, "ratios": [], "trace": [],
        "stop_reason": f"step limit ({max_steps})",
    }  # fmt: skip
    config = {"configurable": {"thread_id": thread_id or uuid.uuid4().hex},
              "recursion_limit": 2 * max_steps + 5}  # fmt: skip
    final = graph.invoke(start, config)
    return AgentResult(
        answer=Answer(**final["answer"]),
        trace=[Step(**s) for s in final["trace"]],
        stop_reason=final["stop_reason"],
        untraced_numbers=final["untraced"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the LangGraph agent a question.")
    parser.add_argument("question")
    parser.add_argument("--model", default=None, help="override LLM_MODEL")
    args = parser.parse_args()
    settings = Settings()
    if args.model:
        settings = settings.model_copy(update={"llm_model": args.model})
    runner = ToolRunner(load_default_search(settings), settings.facts_db)
    thread_id = uuid.uuid4().hex[:12]
    print(f"Q: {args.question}  (thread {thread_id})")
    path = settings.data_dir / "agent_checkpoints.sqlite"
    with SqliteSaver.from_conn_string(str(path)) as saver:
        result = run_graph(args.question, LLMClient(settings), runner, on_step=print_step,
                           checkpointer=saver, thread_id=thread_id)  # fmt: skip
    a = result.answer
    print(f"\nA: {a.answer}")
    for c in a.citations:
        print(f"  [{c.kind}] {c.ref}: {c.quote_or_row[:140]}")
    print(f"\nstop: {result.stop_reason} | LLM calls: {a.steps} | tokens: {a.tokens:,}")
    print(f"dropped citations: {a.dropped_citations} | untraced numbers: {result.untraced_numbers}")
    print(f"checkpoints saved in {path.name}, thread {thread_id}")


if __name__ == "__main__":
    main()

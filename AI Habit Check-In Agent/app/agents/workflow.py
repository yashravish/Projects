from typing import TypedDict
from langgraph.graph import StateGraph, END
from app.schemas.checkin import CheckInRequest, CoachOutput
from app.schemas.evaluation import EvaluationOutput
from app.agents.coach_agent import run_coach_agent
from app.agents.evaluator_agent import run_evaluator_agent
from app.utils.logging import logger


class WorkflowState(TypedDict):
    """State passed between nodes in the LangGraph workflow."""
    request: CheckInRequest
    coach_output: CoachOutput | None
    evaluation: EvaluationOutput | None


async def coach_node(state: WorkflowState) -> dict:
    """Node 1: Generate coaching response from user input."""
    logger.info("Workflow: entering coach node")
    coach_output = await run_coach_agent(state["request"])
    return {"coach_output": coach_output}


async def evaluator_node(state: WorkflowState) -> dict:
    """Node 2: Evaluate the coaching response quality."""
    logger.info("Workflow: entering evaluator node")
    evaluation = await run_evaluator_agent(state["coach_output"])
    return {"evaluation": evaluation}


def build_workflow() -> StateGraph:
    """Build the 2-node LangGraph workflow: Coach -> Evaluator."""
    workflow = StateGraph(WorkflowState)

    workflow.add_node("coach", coach_node)
    workflow.add_node("evaluator", evaluator_node)

    workflow.set_entry_point("coach")
    workflow.add_edge("coach", "evaluator")
    workflow.add_edge("evaluator", END)

    return workflow


# Pre-compiled workflow graph for reuse
compiled_workflow = build_workflow().compile()


async def run_checkin_workflow(
    request: CheckInRequest,
) -> tuple[CoachOutput, EvaluationOutput]:
    """Execute the full check-in workflow and return coach output + evaluation."""
    logger.info("Starting check-in workflow")

    initial_state: WorkflowState = {
        "request": request,
        "coach_output": None,
        "evaluation": None,
    }

    result = await compiled_workflow.ainvoke(initial_state)

    logger.info("Check-in workflow completed successfully")
    return result["coach_output"], result["evaluation"]

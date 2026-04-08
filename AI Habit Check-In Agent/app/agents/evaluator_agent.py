import json
from openai import AsyncOpenAI
from app.config import get_settings
from app.schemas.checkin import CoachOutput
from app.schemas.evaluation import EvaluationOutput
from app.utils.logging import logger

SYSTEM_PROMPT = """You are a quality evaluator for health coaching responses. 
You assess coaching outputs on four dimensions, scoring each from 1 to 10.

Scoring criteria:
- actionability: Is the advice practical and easy to act on? (1=vague, 10=very specific and doable)
- empathy: Is the tone warm, understanding, and supportive? (1=cold/robotic, 10=deeply empathetic)
- specificity: Is the response personalized to the user's input? (1=generic, 10=highly tailored)
- safety: Is the advice safe and responsible? Does it avoid medical claims? (1=risky, 10=completely safe)

Also provide brief overall notes (1-2 sentences) summarizing your evaluation.

Respond with a JSON object with exactly these keys:
- "actionability": integer 1-10
- "empathy": integer 1-10
- "specificity": integer 1-10
- "safety": integer 1-10
- "overall_notes": string

Respond ONLY with valid JSON. No markdown, no extra text."""


async def run_evaluator_agent(coach_output: CoachOutput) -> EvaluationOutput:
    """Score the coaching output on actionability, empathy, specificity, and safety."""
    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    coaching_text = (
        f"Summary: {coach_output.summary}\n"
        f"Habit Risk: {coach_output.habit_risk}\n"
        f"Next Action: {coach_output.next_action}\n"
        f"Motivational Message: {coach_output.motivational_message}"
    )

    logger.info("Calling evaluator agent LLM")
    response = await client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": coaching_text},
        ],
        temperature=0.3,
        max_tokens=300,
        response_format={"type": "json_object"},
    )

    raw_content = response.choices[0].message.content
    logger.info("Evaluator agent LLM response received")

    parsed = json.loads(raw_content)
    return EvaluationOutput(**parsed)

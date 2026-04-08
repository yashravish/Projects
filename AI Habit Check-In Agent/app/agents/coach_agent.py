import json
import re
from openai import AsyncOpenAI
from app.config import get_settings
from app.schemas.checkin import CoachOutput, CheckInRequest
from app.utils.logging import logger

# Crisis keywords that trigger a safe fallback response
CRISIS_KEYWORDS = [
    "suicide", "self-harm", "kill myself", "want to die",
    "end my life", "hurt myself", "not worth living",
]

SAFE_FALLBACK = CoachOutput(
    summary="I appreciate you sharing how you feel. Your well-being matters deeply, and I want to make sure you get the right support.",
    habit_risk="The feelings you described suggest it would be valuable to talk with a professional who can help.",
    next_action="Please reach out to the 988 Suicide & Crisis Lifeline by calling or texting 988, or contact a trusted person in your life.",
    motivational_message="You are not alone, and asking for help is a sign of incredible strength. Support is available 24/7.",
)

SYSTEM_PROMPT = """You are a supportive digital health coach. Your role is to provide brief, 
practical, and encouraging feedback based on a user's health check-in.

STRICT RULES:
- Never provide medical diagnoses
- Never suggest extreme diets or dangerous exercise routines
- Keep outputs short, supportive, and behavior-focused
- Prefer practical, low-risk suggestions
- Focus on small, sustainable behavior changes

Respond with a JSON object with exactly these keys:
- "summary": A short personalized coaching response (2-3 sentences)
- "habit_risk": One habit risk or pattern you identified
- "next_action": One specific actionable next step
- "motivational_message": A supportive motivational message (1-2 sentences)

Respond ONLY with valid JSON. No markdown, no extra text."""


def _contains_crisis_language(text: str) -> bool:
    """Check if any input field contains crisis-related keywords."""
    combined = text.lower()
    return any(keyword in combined for keyword in CRISIS_KEYWORDS)


async def run_coach_agent(request: CheckInRequest) -> CoachOutput:
    """Generate coaching output from user check-in data.
    
    Returns a safe fallback response if crisis language is detected.
    """
    combined_input = f"{request.health_goal} {request.todays_actions} {request.current_mood}"

    if _contains_crisis_language(combined_input):
        logger.warning("CRISIS LANGUAGE DETECTED in check-in input. Returning safe fallback.")
        return SAFE_FALLBACK

    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    user_message = (
        f"Health Goal: {request.health_goal}\n"
        f"Today's Actions: {request.todays_actions}\n"
        f"Current Mood: {request.current_mood}"
    )

    logger.info("Calling coach agent LLM")
    response = await client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
        max_tokens=500,
        response_format={"type": "json_object"},
    )

    raw_content = response.choices[0].message.content
    logger.info("Coach agent LLM response received")

    parsed = json.loads(raw_content)
    return CoachOutput(**parsed)

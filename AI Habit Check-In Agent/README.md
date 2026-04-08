# 🧠 AI Habit Check-In Agent

A backend-only health behavior check-in API powered by AI coaching. Users submit their health goals, daily actions, and mood to receive personalized coaching feedback with quality evaluation — all through a clean REST API.

Built with **FastAPI**, **LangGraph**, **OpenAI**, and **SQLite**.

---

## 📐 Architecture

```
┌─────────────┐     ┌──────────────────────────────────────────┐     ┌──────────┐
│   Client     │────▶│            FastAPI Server                │────▶│  SQLite   │
│  (curl/etc)  │◀────│                                          │◀────│    DB     │
└─────────────┘     │  ┌─────────────────────────────────────┐  │     └──────────┘
                    │  │       LangGraph Workflow             │  │
                    │  │                                       │  │
                    │  │  ┌─────────────┐   ┌──────────────┐  │  │
                    │  │  │ Coach Agent  │──▶│  Evaluator   │  │  │
                    │  │  │  (Node 1)    │   │   Agent      │  │  │
                    │  │  │  OpenAI API  │   │  (Node 2)    │  │  │
                    │  │  └─────────────┘   │  OpenAI API  │  │  │
                    │  │                     └──────────────┘  │  │
                    │  └─────────────────────────────────────┘  │
                    └──────────────────────────────────────────┘
```

### Workflow

1. **User** submits a check-in via `POST /checkins`
2. **Coach Agent** (LangGraph Node 1) generates personalized coaching using OpenAI
3. **Evaluator Agent** (LangGraph Node 2) scores the coaching on actionability, empathy, specificity, and safety
4. Results are **stored in SQLite** and returned as structured JSON

### Safety Features

- Crisis language detection with safe fallback responses (includes 988 Lifeline referral)
- No medical diagnoses or extreme diet advice
- All outputs are behavior-focused and supportive

---

## 🗂 Project Structure

```
ai-habit-checkin-agent/
├── app/
│   ├── main.py                    # FastAPI app entry point
│   ├── config.py                  # Environment-based configuration
│   ├── api/routes.py              # API endpoint definitions
│   ├── agents/
│   │   ├── coach_agent.py         # Coach LLM agent with crisis detection
│   │   ├── evaluator_agent.py     # Evaluator LLM agent for scoring
│   │   └── workflow.py            # LangGraph 2-node workflow
│   ├── db/
│   │   ├── database.py            # SQLite connection management
│   │   ├── models.py              # Data model definitions
│   │   └── crud.py                # Create/read database operations
│   ├── schemas/
│   │   ├── checkin.py             # Request/response Pydantic models
│   │   └── evaluation.py          # Evaluation scoring model
│   ├── services/
│   │   └── checkin_service.py     # Business logic orchestration
│   └── utils/
│       └── logging.py             # Centralized logging setup
├── tests/
│   ├── conftest.py                # Shared test fixtures
│   ├── test_health.py             # Health endpoint tests
│   ├── test_checkins.py           # Check-in CRUD tests (mocked LLM)
│   └── test_evaluator_format.py   # Schema validation & crisis detection tests
├── sample_data/
│   └── sample_checkin.json        # Example request payload
├── .github/workflows/ci.yml      # GitHub Actions CI pipeline
├── Dockerfile                     # Container configuration
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variable template
├── pytest.ini                     # Pytest configuration
└── README.md                     # This file
```

---

## 🚀 Setup & Run

### Prerequisites

- Python 3.12+
- An OpenAI API key

### Local Setup

```bash
# Clone the repo
git clone https://github.com/your-username/ai-habit-checkin-agent.git
cd ai-habit-checkin-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your OpenAI API key

# Run the server
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.  
Interactive docs at `http://localhost:8000/docs`.

### Docker

```bash
docker build -t habit-agent .
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-your-key habit-agent
```

---

## 📡 API Endpoints

| Method | Endpoint              | Description                          |
|--------|-----------------------|--------------------------------------|
| GET    | `/health`             | Health check                         |
| POST   | `/checkins`           | Submit a new check-in                |
| GET    | `/checkins/{id}`      | Retrieve a specific check-in         |
| GET    | `/checkins`           | List all check-ins                   |

---

## 📋 Example Requests

### Health Check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "ai-habit-checkin-agent"
}
```

### Submit a Check-In

```bash
curl -X POST http://localhost:8000/checkins \
  -H "Content-Type: application/json" \
  -d '{
    "health_goal": "eat better and lose weight",
    "todays_actions": "I skipped breakfast, had fast food for lunch, and walked 5000 steps",
    "current_mood": "stressed and tired"
  }'
```

**Response:**
```json
{
  "id": 1,
  "health_goal": "eat better and lose weight",
  "todays_actions": "I skipped breakfast, had fast food for lunch, and walked 5000 steps",
  "current_mood": "stressed and tired",
  "coach_output": {
    "summary": "Great job hitting 5000 steps today! Stress can make food choices harder, so let's focus on one small win for tomorrow.",
    "habit_risk": "Skipping breakfast may lead to overeating later in the day, especially when stress is high.",
    "next_action": "Tonight, prepare a simple overnight oats jar — it takes 3 minutes and gives you a healthy grab-and-go breakfast.",
    "motivational_message": "Progress isn't about perfection. Those 5000 steps prove you're already moving in the right direction!"
  },
  "evaluation": {
    "actionability": 8,
    "empathy": 9,
    "specificity": 7,
    "safety": 10,
    "overall_notes": "Strong empathetic tone with a specific, practical breakfast suggestion. Could personalize more to the lunch situation."
  },
  "created_at": "2026-04-08T18:30:00.000000"
}
```

### Get a Check-In by ID

```bash
curl http://localhost:8000/checkins/1
```

### List All Check-Ins

```bash
curl http://localhost:8000/checkins
```

---

## 🧪 Testing

Tests mock all LLM calls so they run without an API key.

```bash
# Run all tests
pytest -v

# Run with coverage
pytest --cov=app --cov-report=term-missing

# Run specific test file
pytest tests/test_health.py -v
```

---

## 🏗 CI/CD

GitHub Actions runs on every push and PR to `main`:
- Installs Python 3.12 and dependencies
- Runs the full test suite
- Reports code coverage

See [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

---

## ⚖️ Tradeoffs & Design Decisions

| Decision | Rationale |
|----------|-----------|
| **SQLite over PostgreSQL** | Simplicity for an MVP — zero config, file-based. Easy to swap later. |
| **aiosqlite over SQLAlchemy** | Lightweight async access without ORM overhead for simple schema. |
| **Mocked tests** | Tests don't require a live API key, making CI fast and free. |
| **LangGraph for 2 nodes** | Slight over-engineering for 2 nodes, but demonstrates the pattern for future multi-agent expansion. |
| **`json_object` response format** | Forces structured JSON from OpenAI, eliminating fragile regex parsing. |
| **Crisis detection via keyword matching** | Simple but effective first pass. A production system would use a classifier. |
| **No authentication** | Intentionally omitted for MVP scope. Would add JWT/API key auth in production. |
| **Single model for both agents** | Uses the same model (configurable). Could use a cheaper model for evaluation. |

---

## 🔮 Future Improvements

- [ ] **Authentication** — Add API key or JWT-based auth
- [ ] **Streak tracking** — Track consecutive check-in days
- [ ] **Trend analysis** — Analyze mood and habit patterns over time
- [ ] **Multi-model routing** — Use different models per agent (e.g., GPT-4 for coaching, GPT-4o-mini for eval)
- [ ] **Rate limiting** — Prevent API abuse
- [ ] **PostgreSQL migration** — For production scalability
- [ ] **Webhook notifications** — Send daily reminders
- [ ] **More sophisticated crisis detection** — ML-based classifier instead of keyword matching
- [ ] **Response caching** — Cache similar check-in patterns
- [ ] **Admin dashboard** — View aggregate stats and flagged check-ins

---

## 📸 Screenshots

> _Screenshots will be added after UI development or API documentation generation._

| Screenshot | Description |
|------------|-------------|
| ![API Docs](screenshots/api-docs.png) | FastAPI Swagger UI |
| ![Check-In Response](screenshots/checkin-response.png) | Example check-in JSON response |

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

import hashlib

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.ai_report import AIAnalysisReport, ReportSeverity
from app.models.defect import Defect, DefectPriority, DefectSeverity, DefectStatus
from app.models.knowledge_base import KnowledgeBaseArticle
from app.schemas.ai import AIAnalysisRead, AIAnalyzeRequest
from app.services.ai_analyzer import analyze_failure_logs


router = APIRouter(prefix="/api/ai", tags=["ai"])


@router.post("/analyze", response_model=AIAnalysisRead, status_code=201)
async def analyze(
    body: AIAnalyzeRequest, session: AsyncSession = Depends(get_db)
) -> AIAnalysisRead:
    result = analyze_failure_logs(body.logs)
    h = hashlib.sha256(body.logs.encode("utf-8", errors="replace")).hexdigest()
    try:
        sev = ReportSeverity(result.severity)
    except Exception:
        sev = ReportSeverity.medium
    r = AIAnalysisReport(
        log_snippet=body.logs[:16_000],
        log_hash=h,
        root_cause_summary=result.root_cause_summary,
        likely_file_or_component=result.likely_file_or_component,
        suggested_fix=result.suggested_fix,
        severity=sev,
        confidence_score=result.confidence_score,
        project_id=body.project_id,
    )
    session.add(r)
    await session.flush()
    if body.link_kb_article_ids:
        for kid in set(body.link_kb_article_ids):
            a = await session.get(KnowledgeBaseArticle, kid)
            if a:
                r.linked_articles.append(a)
    defect: Defect | None = None
    if body.create_defect and body.project_id is not None:
        defect = Defect(
            project_id=body.project_id,
            title=f"AI: {result.root_cause_summary[:80]}",
            description=body.logs[:2000],
            severity=DefectSeverity(sev.value),
            priority=DefectPriority.p1 if sev in (ReportSeverity.high, ReportSeverity.critical) else DefectPriority.p2,
            status=DefectStatus.open,
            root_cause=result.root_cause_summary,
            suggested_fix=result.suggested_fix,
            ai_report_id=r.id,
        )
        session.add(defect)
        for kid in set(body.link_kb_article_ids or []):
            a = await session.get(KnowledgeBaseArticle, kid)
            if a:
                defect.linked_articles.append(a)
    await session.flush()
    await session.refresh(r, attribute_names=["linked_articles", "created_defect"])
    return _read_report(session, r, defect_id=defect.id if defect else None)


def _read_report(session: AsyncSession, r: AIAnalysisReport, defect_id: int | None) -> AIAnalysisRead:
    ids = [a.id for a in (r.linked_articles or [])]
    did = defect_id
    if not did and r.created_defect:
        did = r.created_defect.id
    return AIAnalysisRead(
        id=r.id,
        log_hash=r.log_hash,
        root_cause_summary=r.root_cause_summary,
        likely_file_or_component=r.likely_file_or_component,
        suggested_fix=r.suggested_fix,
        severity=r.severity.value,
        confidence_score=r.confidence_score,
        project_id=r.project_id,
        created_at=r.created_at,
        created_defect_id=did,
        linked_kb_article_ids=ids,
    )

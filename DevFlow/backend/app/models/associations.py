from sqlalchemy import Column, ForeignKey, Integer, Table

from app.core.database import Base

defect_kb_article = Table(
    "defect_kb_article",
    Base.metadata,
    Column("defect_id", Integer, ForeignKey("defects.id", ondelete="CASCADE"), primary_key=True),
    Column("article_id", Integer, ForeignKey("knowledge_base_articles.id", ondelete="CASCADE"), primary_key=True),
)

analysis_kb_article = Table(
    "analysis_kb_article",
    Base.metadata,
    Column(
        "analysis_id",
        Integer,
        ForeignKey("ai_analysis_reports.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("article_id", Integer, ForeignKey("knowledge_base_articles.id", ondelete="CASCADE"), primary_key=True),
)

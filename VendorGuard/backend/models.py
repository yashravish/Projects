import json
from datetime import datetime, date
from typing import Optional, List

from sqlalchemy import (
    Column, Integer, String, Text, Boolean, Float, Date, DateTime,
    ForeignKey, func,
)
from sqlalchemy.orm import relationship, Mapped, mapped_column

from backend.database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str] = mapped_column(String(255), default="")
    role: Mapped[str] = mapped_column(String(50), default="analyst")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    audit_logs = relationship("AuditLog", back_populates="user")


class Vendor(Base):
    __tablename__ = "vendors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    category: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, default="")
    website: Mapped[Optional[str]] = mapped_column(String(500), default="")
    business_owner: Mapped[Optional[str]] = mapped_column(String(255), default="")
    vendor_contact: Mapped[Optional[str]] = mapped_column(String(255), default="")
    hosting_model: Mapped[Optional[str]] = mapped_column(String(100), default="")
    deployment_scope: Mapped[Optional[str]] = mapped_column(String(100), default="")
    internet_exposed: Mapped[bool] = mapped_column(Boolean, default=False)
    handles_sensitive_data: Mapped[bool] = mapped_column(Boolean, default=False)
    data_types_json: Mapped[Optional[str]] = mapped_column(Text, default="[]")
    compliance_attestations_json: Mapped[Optional[str]] = mapped_column(Text, default="[]")
    status: Mapped[str] = mapped_column(String(50), default="active")
    created_by: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    integrations = relationship("VendorIntegration", back_populates="vendor", cascade="all, delete-orphan")
    assessments = relationship("Assessment", back_populates="vendor", cascade="all, delete-orphan")

    @property
    def data_types(self) -> list:
        try:
            return json.loads(self.data_types_json or "[]")
        except (json.JSONDecodeError, TypeError):
            return []

    @data_types.setter
    def data_types(self, value: list):
        self.data_types_json = json.dumps(value)

    @property
    def compliance_attestations(self) -> list:
        try:
            return json.loads(self.compliance_attestations_json or "[]")
        except (json.JSONDecodeError, TypeError):
            return []

    @compliance_attestations.setter
    def compliance_attestations(self, value: list):
        self.compliance_attestations_json = json.dumps(value)


class VendorIntegration(Base):
    __tablename__ = "vendor_integrations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    vendor_id: Mapped[int] = mapped_column(Integer, ForeignKey("vendors.id"), nullable=False)
    system_name: Mapped[str] = mapped_column(String(255), nullable=False)
    integration_type: Mapped[str] = mapped_column(String(100), default="")
    data_flow_direction: Mapped[str] = mapped_column(String(50), default="bidirectional")
    description: Mapped[Optional[str]] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    vendor = relationship("Vendor", back_populates="integrations")


class ControlDomain(Base):
    __tablename__ = "control_domains"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(10), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, default="")
    nist_mapping: Mapped[Optional[str]] = mapped_column(String(255), default="")
    iso_mapping: Mapped[Optional[str]] = mapped_column(String(255), default="")


class Assessment(Base):
    __tablename__ = "assessments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    vendor_id: Mapped[int] = mapped_column(Integer, ForeignKey("vendors.id"), nullable=False)
    assessment_type: Mapped[str] = mapped_column(String(100), default="initial")
    phase: Mapped[str] = mapped_column(String(50), default="pre_implementation")
    assessor_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id"), nullable=True)
    overall_inherent_risk: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    inherent_risk_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    overall_residual_risk: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    residual_risk_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="draft")
    executive_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    ai_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    vendor = relationship("Vendor", back_populates="assessments")
    assessor = relationship("User", foreign_keys=[assessor_id])
    answers = relationship("AssessmentAnswer", back_populates="assessment", cascade="all, delete-orphan")
    findings = relationship("Finding", back_populates="assessment", cascade="all, delete-orphan")
    reports = relationship("GeneratedReport", back_populates="assessment", cascade="all, delete-orphan")


class AssessmentAnswer(Base):
    __tablename__ = "assessment_answers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    assessment_id: Mapped[int] = mapped_column(Integer, ForeignKey("assessments.id"), nullable=False)
    question_key: Mapped[str] = mapped_column(String(255), nullable=False)
    section: Mapped[str] = mapped_column(String(100), default="")
    question_text: Mapped[Optional[str]] = mapped_column(Text, default="")
    answer: Mapped[Optional[str]] = mapped_column(Text, default="")
    notes: Mapped[Optional[str]] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    assessment = relationship("Assessment", back_populates="answers")


class Finding(Base):
    __tablename__ = "findings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    assessment_id: Mapped[int] = mapped_column(Integer, ForeignKey("assessments.id"), nullable=False)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, default="")
    severity: Mapped[str] = mapped_column(String(50), default="Low")
    likelihood: Mapped[str] = mapped_column(String(50), default="Moderate")
    impact: Mapped[str] = mapped_column(String(50), default="Moderate")
    control_domain_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("control_domains.id"), nullable=True)
    recommendation: Mapped[Optional[str]] = mapped_column(Text, default="")
    owner: Mapped[Optional[str]] = mapped_column(String(255), default="")
    due_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    remediation_status: Mapped[str] = mapped_column(String(50), default="open")
    source_rule: Mapped[Optional[str]] = mapped_column(String(255), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    assessment = relationship("Assessment", back_populates="findings")
    control_domain = relationship("ControlDomain")
    remediation_items = relationship("RemediationItem", back_populates="finding", cascade="all, delete-orphan")


class RemediationItem(Base):
    __tablename__ = "remediation_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    finding_id: Mapped[int] = mapped_column(Integer, ForeignKey("findings.id"), nullable=False)
    action: Mapped[str] = mapped_column(Text, nullable=False)
    assigned_to: Mapped[Optional[str]] = mapped_column(String(255), default="")
    priority: Mapped[str] = mapped_column(String(50), default="Medium")
    status: Mapped[str] = mapped_column(String(50), default="open")
    due_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    completion_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    finding = relationship("Finding", back_populates="remediation_items")


class AssessmentTemplate(Base):
    __tablename__ = "assessment_templates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    category: Mapped[str] = mapped_column(String(100), default="general")
    description: Mapped[Optional[str]] = mapped_column(Text, default="")
    questions_json: Mapped[Optional[str]] = mapped_column(Text, default="[]")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    @property
    def questions(self) -> list:
        try:
            return json.loads(self.questions_json or "[]")
        except (json.JSONDecodeError, TypeError):
            return []

    @questions.setter
    def questions(self, value: list):
        self.questions_json = json.dumps(value)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id"), nullable=True)
    action: Mapped[str] = mapped_column(String(255), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(100), default="")
    entity_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    details: Mapped[Optional[str]] = mapped_column(Text, default="")
    ip_address: Mapped[Optional[str]] = mapped_column(String(50), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    user = relationship("User", back_populates="audit_logs")


class GeneratedReport(Base):
    __tablename__ = "generated_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    assessment_id: Mapped[int] = mapped_column(Integer, ForeignKey("assessments.id"), nullable=False)
    report_type: Mapped[str] = mapped_column(String(100), default="full")
    file_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    generated_by: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    assessment = relationship("Assessment", back_populates="reports")
    generator = relationship("User", foreign_keys=[generated_by])

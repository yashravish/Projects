from datetime import datetime, date
from typing import Optional, List
from pydantic import BaseModel, Field


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class LoginRequest(BaseModel):
    username: str
    password: str


class UserOut(BaseModel):
    id: int
    username: str
    email: str
    full_name: str
    role: str
    is_active: bool

    model_config = {"from_attributes": True}


class VendorCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    category: str = Field(..., min_length=1)
    description: str = ""
    website: str = ""
    business_owner: str = ""
    vendor_contact: str = ""
    hosting_model: str = ""
    deployment_scope: str = ""
    internet_exposed: bool = False
    handles_sensitive_data: bool = False
    data_types: List[str] = []
    compliance_attestations: List[str] = []
    status: str = "active"


class VendorOut(BaseModel):
    id: int
    name: str
    category: str
    description: Optional[str] = ""
    website: Optional[str] = ""
    business_owner: Optional[str] = ""
    vendor_contact: Optional[str] = ""
    hosting_model: Optional[str] = ""
    deployment_scope: Optional[str] = ""
    internet_exposed: bool
    handles_sensitive_data: bool
    data_types: list = []
    compliance_attestations: list = []
    status: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class IntegrationCreate(BaseModel):
    system_name: str
    integration_type: str = ""
    data_flow_direction: str = "bidirectional"
    description: str = ""


class IntegrationOut(BaseModel):
    id: int
    vendor_id: int
    system_name: str
    integration_type: str
    data_flow_direction: str
    description: str

    model_config = {"from_attributes": True}


class AssessmentCreate(BaseModel):
    vendor_id: int
    assessment_type: str = "initial"
    phase: str = "pre_implementation"


class AnswerSubmission(BaseModel):
    question_key: str
    section: str
    question_text: str = ""
    answer: str
    notes: str = ""


class AssessmentSubmit(BaseModel):
    answers: List[AnswerSubmission]


class FindingOut(BaseModel):
    id: int
    assessment_id: int
    title: str
    description: Optional[str] = ""
    severity: str
    likelihood: str
    impact: str
    control_domain_id: Optional[int] = None
    control_domain_name: Optional[str] = None
    recommendation: Optional[str] = ""
    owner: Optional[str] = ""
    due_date: Optional[date] = None
    remediation_status: str
    source_rule: Optional[str] = ""
    created_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class AssessmentOut(BaseModel):
    id: int
    vendor_id: int
    vendor_name: Optional[str] = ""
    assessment_type: str
    phase: str
    assessor_id: Optional[int] = None
    overall_inherent_risk: Optional[str] = None
    inherent_risk_score: Optional[float] = None
    overall_residual_risk: Optional[str] = None
    residual_risk_score: Optional[float] = None
    status: str
    executive_summary: Optional[str] = None
    ai_summary: Optional[str] = None
    findings_count: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class RemediationUpdate(BaseModel):
    assigned_to: Optional[str] = None
    status: Optional[str] = None
    due_date: Optional[date] = None
    notes: Optional[str] = None


class RemediationOut(BaseModel):
    id: int
    finding_id: int
    finding_title: Optional[str] = ""
    vendor_name: Optional[str] = ""
    action: str
    assigned_to: Optional[str] = ""
    priority: str
    status: str
    due_date: Optional[date] = None
    completion_date: Optional[date] = None
    notes: Optional[str] = ""
    created_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class ControlDomainOut(BaseModel):
    id: int
    code: str
    name: str
    description: Optional[str] = ""
    nist_mapping: Optional[str] = ""
    iso_mapping: Optional[str] = ""

    model_config = {"from_attributes": True}


class DashboardStats(BaseModel):
    total_vendors: int = 0
    active_assessments: int = 0
    open_critical_findings: int = 0
    open_high_findings: int = 0
    overdue_remediations: int = 0
    vendors_by_category: dict = {}
    findings_by_severity: dict = {}
    findings_by_domain: dict = {}
    recent_activity: list = []

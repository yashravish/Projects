import json
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.auth import get_current_user
from backend.models import User, Vendor, VendorIntegration
from backend.schemas import VendorCreate, VendorOut, IntegrationCreate, IntegrationOut
from backend.services.audit_service import log_action

router = APIRouter(prefix="/api/vendors", tags=["vendors"])

VALID_CATEGORIES = [
    "SaaS", "PaaS", "AI Tool", "IoT Platform",
    "Tokenization Platform", "Distributed Ledger Platform",
    "End-User Software Package",
]


@router.get("", response_model=list[VendorOut])
def list_vendors(
    status: str | None = Query(None),
    category: str | None = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    q = db.query(Vendor)
    if status:
        q = q.filter(Vendor.status == status)
    if category:
        q = q.filter(Vendor.category == category)
    vendors = q.order_by(Vendor.created_at.desc()).all()
    result = []
    for v in vendors:
        out = VendorOut.model_validate(v)
        out.data_types = v.data_types
        out.compliance_attestations = v.compliance_attestations
        result.append(out)
    return result


@router.post("", response_model=VendorOut, status_code=201)
def create_vendor(
    body: VendorCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if body.category not in VALID_CATEGORIES:
        raise HTTPException(status_code=400, detail=f"Invalid category. Must be one of: {VALID_CATEGORIES}")
    vendor = Vendor(
        name=body.name,
        category=body.category,
        description=body.description,
        website=body.website,
        business_owner=body.business_owner,
        vendor_contact=body.vendor_contact,
        hosting_model=body.hosting_model,
        deployment_scope=body.deployment_scope,
        internet_exposed=body.internet_exposed,
        handles_sensitive_data=body.handles_sensitive_data,
        data_types_json=json.dumps(body.data_types),
        compliance_attestations_json=json.dumps(body.compliance_attestations),
        status=body.status,
        created_by=current_user.id,
    )
    db.add(vendor)
    db.commit()
    db.refresh(vendor)
    log_action(db, current_user.id, "vendor_created", "vendor", vendor.id, details=vendor.name)
    out = VendorOut.model_validate(vendor)
    out.data_types = vendor.data_types
    out.compliance_attestations = vendor.compliance_attestations
    return out


@router.get("/{vendor_id}", response_model=VendorOut)
def get_vendor(vendor_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    vendor = db.query(Vendor).get(vendor_id)
    if not vendor:
        raise HTTPException(status_code=404, detail="Vendor not found")
    out = VendorOut.model_validate(vendor)
    out.data_types = vendor.data_types
    out.compliance_attestations = vendor.compliance_attestations
    return out


@router.post("/{vendor_id}/integrations", response_model=IntegrationOut, status_code=201)
def add_integration(
    vendor_id: int,
    body: IntegrationCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    vendor = db.query(Vendor).get(vendor_id)
    if not vendor:
        raise HTTPException(status_code=404, detail="Vendor not found")
    integration = VendorIntegration(
        vendor_id=vendor_id,
        system_name=body.system_name,
        integration_type=body.integration_type,
        data_flow_direction=body.data_flow_direction,
        description=body.description,
    )
    db.add(integration)
    db.commit()
    db.refresh(integration)
    return integration


@router.get("/{vendor_id}/integrations", response_model=list[IntegrationOut])
def list_integrations(
    vendor_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return db.query(VendorIntegration).filter(VendorIntegration.vendor_id == vendor_id).all()

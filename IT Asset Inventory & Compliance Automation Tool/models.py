# models.py
class Asset(Base):
    __tablename__ = "assets"
    id = Column(UUID, primary_key=True)
    asset_type = Column(Enum('Hardware','Software','License'))
    name = Column(String)
    serial_number = Column(String)
    azure_ad_device_id = Column(String)  # For Intune sync
    purchase_date = Column(Date)
    warranty_expiry = Column(Date)
    lifecycle_status = Column(Enum('Active','Retired','E-Waste'))
    
class ComplianceCheck(Base):
    __tablename__ = "compliance_checks"
    id = Column(UUID, primary_key=True)
    asset_id = Column(UUID, ForeignKey('assets.id'))
    check_type = Column(Enum('OS','Antivirus','License'))
    status = Column(Enum('Compliant','Non-Compliant'))
    last_checked = Column(DateTime)

class SoftwareLicense(Base):
    __tablename__ = "licenses"
    id = Column(UUID, primary_key=True)
    product_name = Column(String)
    key = Column(String)
    seats = Column(Integer)
    expiration_date = Column(Date)
    assigned_to = Column(UUID, ForeignKey('assets.id'))
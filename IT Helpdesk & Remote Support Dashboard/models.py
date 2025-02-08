# models.py
class Ticket(Base):
    __tablename__ = "tickets"
    id = Column(UUID, primary_key=True)
    title = Column(String)
    description = Column(String)
    status = Column(Enum('Open','In Progress','Resolved'))
    created_at = Column(DateTime)
    created_by = Column(String)  # AD username
    assigned_to = Column(String) # AD username

class Device(Base):
    __tablename__ = "devices"
    id = Column(UUID, primary_key=True)
    hostname = Column(String)
    type = Column(Enum('Laptop','Desktop','Printer'))
    azure_ad_device_id = Column(String)  # For Intune integration
    last_user = Column(String)
    warranty_expiry = Column(Date)

# auth.py
async def authenticate_ad(username: str, password: str):
    server = Server('ldap://your-ad-server', get_info=ALL)
    conn = Connection(server, user=f"{username}@domain", password=password)
    if conn.bind():
        return get_user_roles(username)  # Query AD group membership
    return None
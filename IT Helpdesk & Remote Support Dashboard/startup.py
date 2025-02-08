@app.on_event("startup")
@repeat_every(seconds=60 * 60 * 24)  # Daily check
def check_warranty_expirations():
    devices = db.query(Device).filter(
        Device.warranty_expiry < datetime.now() + timedelta(days=30)
    ).all()
    send_alert_email(devices)
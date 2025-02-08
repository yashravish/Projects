# api/routers/assets.py
@app.post("/asset-discovery", tags=["Assets"])
async def receive_asset_data(data: AssetDiscoverySchema):
    existing = await check_existing_asset(data.serial_number)
    if existing:
        return await update_asset(existing.id, data)
    else:
        return await create_asset(data)

# api/routers/compliance.py
@app.get("/compliance-report/{asset_id}", tags=["Compliance"])
async def generate_compliance_report(asset_id: UUID):
    asset = await get_asset(asset_id)
    checks = [
        check_os_compliance(asset),
        check_antivirus_status(asset),
        check_license_expiry(asset)
    ]
    return {"compliance_status": all(checks)}
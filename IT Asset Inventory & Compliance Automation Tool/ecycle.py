# api/routers/ecycle.py
@app.post("/flag-e-waste", tags=["Compliance"])
async def flag_e_waste():
    criteria = {
        'warranty_expiry': {'lt': datetime.now()},
        'lifecycle_status': 'Active',
        'age_months': {'gte': 36}
    }
    
    e_waste_candidates = await query_assets(criteria)
    for asset in e_waste_candidates:
        await update_asset(asset.id, {"lifecycle_status": "E-Waste"})
        create_disposal_ticket(asset)
    
    return {"flagged": len(e_waste_candidates)}


# api/compliance/checks.py
async def check_os_compliance(asset):
    baseline = {
        'Windows': '10.0.19044',
        'Linux': 'Ubuntu 20.04'
    }
    
    return asset.os_version >= baseline.get(asset.os_type, "Unknown")

async def generate_audit_report():
    report_data = {
        'assets': await get_asset_summary(),
        'compliance_stats': await get_compliance_stats(),
        'license_status': await get_license_status()
    }
    
    return generate_pdf_report(report_data)
# intune_integration.py
async def get_intune_device_compliance(device_id):
    headers = {
        "Authorization": f"Bearer {get_msal_token()}"
    }
    response = await httpx.get(
        f"https://graph.microsoft.com/v1.0/deviceManagement/managedDevices/{device_id}",
        headers=headers
    )
    return response.json()
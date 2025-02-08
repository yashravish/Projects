# api/clients/azure.py
async def sync_intune_devices():
    graph_client = get_graph_client()
    devices = graph_client.devices.list()
    
    async for device in devices:
        upsert_asset({
            'azure_id': device.id,
            'name': device.display_name,
            'asset_type': 'Hardware',
            'os_version': device.operating_system,
            'last_sync': datetime.utcnow()
        })
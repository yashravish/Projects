def test_asset_discovery_flow():
    test_payload = generate_mock_discovery_data()
    response = client.post("/asset-discovery", json=test_payload)
    assert response.status_code == 200
    assert Asset.objects.count() == 1
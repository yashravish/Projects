def test_e_waste_identification():
    create_old_asset(warranty_expiry=date(2020,1,1))
    response = client.post("/flag-e-waste")
    assert response.json()["flagged"] > 0
# tests/test_incident_handling.py
def test_phishing_incident_workflow():
    report_phishing_email()
    assert Incident.objects.count() == 1
    incident = Incident.objects.first()
    incident.resolve()
    assert incident.status == "Resolved"
# openvas_integration/scanner.py
def trigger_openvas_scan(target_ip):
    conn = OpenVAS_API(config.OPENVAS_USER, config.OPENVAS_PASSWORD)
    scan_id = conn.create_target(target_ip).create_task().start_scan()
    return poll_scan_results(scan_id)

def process_vulnerabilities(scan_results):
    for vuln in scan_results.vulnerabilities:
        SecurityIncident.objects.create(
            title=f"Vulnerability: {vuln.name}",
            incident_type='VULN',
            severity=vuln.threat_level,
            detection_source='OpenVAS'
        )
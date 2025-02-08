# incident/models.py
class SecurityIncident(models.Model):
    INCIDENT_TYPES = [
        ('PHISH', 'Phishing Attempt'),
        ('MAL', 'Malware Detection'),
        ('DDOS', 'DDoS Attack')
    ]
    
    title = models.CharField(max_length=200)
    incident_type = models.CharField(max_length=5, choices=INCIDENT_TYPES)
    severity = models.IntegerField(choices=[(1,'Low'), (2,'Medium'), (3,'High')])
    detection_source = models.CharField(max_length=100)  # OpenVAS, SIEM, etc.
    affected_assets = models.ManyToManyField('Asset')
    status = models.CharField(max_length=20, default='Open')
    created_at = models.DateTimeField(auto_now_add=True)
    resolved_at = models.DateTimeField(null=True)

# patchmgmt/models.py
class PatchSchedule(models.Model):
    target_systems = models.JSONField()  # {"os": "Windows", "versions": ["10.0.1"]}
    patch_kb = models.CharField(max_length=20)  # KB123456
    deployment_status = models.CharField(max_length=20)
    scheduled_time = models.DateTimeField()
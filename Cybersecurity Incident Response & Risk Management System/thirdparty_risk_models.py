# thirdparty_risk/models.py
class VendorQuestionnaire(models.Model):
    VSAF_TEMPLATE = {
        "data_handling": {
            "question": "How is customer data stored?",
            "type": "multiple_choice",
            "options": ["Encrypted", "Plaintext", "Tokenized"]
        }
    }
    
    vendor = models.ForeignKey('Vendor', on_delete=models.CASCADE)
    responses = models.JSONField()
    risk_score = models.FloatField()
    last_audited = models.DateField()

def calculate_risk_score(responses):
    # Implement NIST CSF scoring logic
    return (critical_controls_missing * 0.4) + (data_handling_risk * 0.6)
# incident/signals.py
@receiver(post_save, sender=SecurityIncident)
def trigger_incident_alerts(sender, instance, created, **kwargs):
    if created or instance.status_changed():
        message = f"New {instance.get_incident_type_display()} alert: {instance.title}"
        
        # Slack Integration
        slack_client.chat_postMessage(
            channel="#security-alerts",
            text=message
        )
        
        # Email Teams
        send_mail(
            subject="Security Alert",
            message=message,
            from_email="alerts@company.com",
            recipient_list=security_team_emails()
        )
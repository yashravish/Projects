# Middleware for audit logging
class AuditMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
        
    def __call__(self, request):
        response = self.get_response(request)
        if request.user.is_authenticated:
            AuditLog.objects.create(
                user=request.user,
                action=request.path,
                timestamp=timezone.now()
            )
        return response

# RBAC Configuration
ROLE_PERMISSIONS = {
    'SOC Analyst': ['view_incident', 'update_incident'],
    'CISO': ['view_reports', 'approve_patches'],
    'Vendor Manager': ['complete_questionnaires']
}
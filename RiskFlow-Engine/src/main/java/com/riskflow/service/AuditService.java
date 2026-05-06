package com.riskflow.service;

import com.riskflow.model.AuditLog;
import com.riskflow.repository.AuditLogRepository;
import org.springframework.stereotype.Service;

@Service
public class AuditService {
    private final AuditLogRepository repository;
    public AuditService(AuditLogRepository repository) { this.repository = repository; }
    public void log(String eventType, String entityType, Long entityId, String message) { repository.save(new AuditLog(eventType, entityType, entityId, message)); }
}

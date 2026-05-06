package com.riskflow.model;

import jakarta.persistence.*;
import java.time.Instant;

@Entity
@Table(name = "audit_logs", indexes = @Index(name = "idx_audit_entity_time", columnList = "entityType, entityId, timestamp"))
public class AuditLog {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    @Column(nullable = false) private String eventType;
    @Column(nullable = false) private String entityType;
    @Column(nullable = false) private Long entityId;
    @Column(nullable = false) private String message;
    @Column(nullable = false) private Instant timestamp = Instant.now();
    public AuditLog() {}
    public AuditLog(String eventType, String entityType, Long entityId, String message) { this.eventType = eventType; this.entityType = entityType; this.entityId = entityId; this.message = message; }
    public Long getId() { return id; }
    public String getEventType() { return eventType; }
    public String getEntityType() { return entityType; }
    public Long getEntityId() { return entityId; }
    public String getMessage() { return message; }
    public Instant getTimestamp() { return timestamp; }
}

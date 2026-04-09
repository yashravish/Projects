-- ============================================================
-- Clinical Image Intake Portal - Database Schema
-- Version: 1.0
-- Database: clinical_intake_portal
-- ============================================================

CREATE DATABASE IF NOT EXISTS clinical_intake_portal
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE clinical_intake_portal;

-- ------------------------------------------------------------
-- Users table: stores login credentials and role assignments
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(100) NOT NULL,
    role ENUM('admin', 'support') NOT NULL DEFAULT 'support',
    email VARCHAR(100) NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ------------------------------------------------------------
-- Cases table: core imaging case intake records
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS cases (
    id INT AUTO_INCREMENT PRIMARY KEY,
    patient_first_name VARCHAR(100) NOT NULL,
    patient_last_name VARCHAR(100) NOT NULL,
    date_of_birth DATE NOT NULL,
    clinic_name VARCHAR(150) NOT NULL,
    provider_name VARCHAR(150) NOT NULL,
    imaging_type ENUM('Skin Lesion', 'Facial Analysis', 'Scar Review', 'Follow-Up', 'Other') NOT NULL,
    body_area VARCHAR(100) DEFAULT NULL,
    priority ENUM('Low', 'Medium', 'High', 'Urgent') NOT NULL DEFAULT 'Medium',
    status ENUM('New', 'Under Review', 'Awaiting Clinic Response', 'Verified', 'Escalated', 'Closed') NOT NULL DEFAULT 'New',
    symptoms_notes TEXT DEFAULT NULL,
    image_filename VARCHAR(255) DEFAULT NULL,
    insurance_id VARCHAR(50) DEFAULT NULL,
    patient_email VARCHAR(100) DEFAULT NULL,
    patient_phone VARCHAR(30) DEFAULT NULL,
    assigned_to INT DEFAULT NULL,
    external_sync_status ENUM('not_synced', 'synced', 'failed') NOT NULL DEFAULT 'not_synced',
    external_reference_id VARCHAR(100) DEFAULT NULL,
    soap_verification_status VARCHAR(50) DEFAULT NULL,
    created_by INT DEFAULT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    CONSTRAINT fk_cases_assigned_to FOREIGN KEY (assigned_to) REFERENCES users(id) ON DELETE SET NULL,
    CONSTRAINT fk_cases_created_by FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ------------------------------------------------------------
-- Case status history: audit trail for every status change
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS case_status_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    case_id INT NOT NULL,
    old_status VARCHAR(50) DEFAULT NULL,
    new_status VARCHAR(50) NOT NULL,
    changed_by INT DEFAULT NULL,
    changed_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    notes TEXT DEFAULT NULL,
    CONSTRAINT fk_csh_case FOREIGN KEY (case_id) REFERENCES cases(id) ON DELETE CASCADE,
    CONSTRAINT fk_csh_user FOREIGN KEY (changed_by) REFERENCES users(id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ------------------------------------------------------------
-- Support notes: troubleshooting and escalation notes on cases
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS support_notes (
    id INT AUTO_INCREMENT PRIMARY KEY,
    case_id INT NOT NULL,
    author_id INT DEFAULT NULL,
    note_body TEXT NOT NULL,
    note_type ENUM('support', 'technical', 'customer_issue', 'sync_issue') NOT NULL DEFAULT 'support',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_sn_case FOREIGN KEY (case_id) REFERENCES cases(id) ON DELETE CASCADE,
    CONSTRAINT fk_sn_author FOREIGN KEY (author_id) REFERENCES users(id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ------------------------------------------------------------
-- Integration logs: REST sync attempts to external systems
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS integration_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    case_id INT NOT NULL,
    endpoint VARCHAR(255) NOT NULL,
    request_payload TEXT DEFAULT NULL,
    response_payload TEXT DEFAULT NULL,
    http_status INT DEFAULT NULL,
    success TINYINT(1) NOT NULL DEFAULT 0,
    error_message TEXT DEFAULT NULL,
    attempted_by INT DEFAULT NULL,
    attempted_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_il_case FOREIGN KEY (case_id) REFERENCES cases(id) ON DELETE CASCADE,
    CONSTRAINT fk_il_user FOREIGN KEY (attempted_by) REFERENCES users(id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ------------------------------------------------------------
-- SOAP verifications: clinic/insurance verification results
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS soap_verifications (
    id INT AUTO_INCREMENT PRIMARY KEY,
    case_id INT NOT NULL,
    clinic_name VARCHAR(150) NOT NULL,
    insurance_id VARCHAR(50) NOT NULL,
    verification_status VARCHAR(50) NOT NULL,
    clinic_approved TINYINT(1) NOT NULL DEFAULT 0,
    policy_type VARCHAR(50) DEFAULT NULL,
    message TEXT DEFAULT NULL,
    checked_at DATETIME NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_sv_case FOREIGN KEY (case_id) REFERENCES cases(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ------------------------------------------------------------
-- Application logs: system-level error and event logging
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS app_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    level ENUM('info', 'warning', 'error', 'critical') NOT NULL DEFAULT 'info',
    message TEXT NOT NULL,
    context TEXT DEFAULT NULL,
    file VARCHAR(255) DEFAULT NULL,
    line INT DEFAULT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Add indexes for common queries
CREATE INDEX idx_cases_status ON cases(status);
CREATE INDEX idx_cases_priority ON cases(priority);
CREATE INDEX idx_cases_clinic ON cases(clinic_name);
CREATE INDEX idx_cases_imaging_type ON cases(imaging_type);
CREATE INDEX idx_cases_created_at ON cases(created_at);
CREATE INDEX idx_csh_case_id ON case_status_history(case_id);
CREATE INDEX idx_sn_case_id ON support_notes(case_id);
CREATE INDEX idx_il_case_id ON integration_logs(case_id);
CREATE INDEX idx_sv_case_id ON soap_verifications(case_id);
CREATE INDEX idx_app_logs_level ON app_logs(level);
CREATE INDEX idx_app_logs_created ON app_logs(created_at);

-- ============================================================
-- Clinical Image Intake Portal - Seed Data
-- NOTE: For initial setup, use setup.php which generates
-- proper password hashes. This file is for reference only.
-- ============================================================

USE clinical_intake_portal;

-- ------------------------------------------------------------
-- Users (passwords are hashed via setup.php)
-- Demo credentials:
--   admin   / admin123   (role: admin)
--   sarah   / support123 (role: support)
-- ------------------------------------------------------------
-- INSERT INTO users handled by setup.php with password_hash()

-- ------------------------------------------------------------
-- Sample Cases (12 realistic clinical imaging cases)
-- ------------------------------------------------------------
INSERT INTO cases (patient_first_name, patient_last_name, date_of_birth, clinic_name, provider_name, imaging_type, body_area, priority, status, symptoms_notes, image_filename, insurance_id, patient_email, patient_phone, assigned_to, external_sync_status, external_reference_id, soap_verification_status, created_by, created_at) VALUES
('Maria', 'Santos', '1985-03-15', 'Riverside Medical Center', 'Dr. Angela Cruz', 'Skin Lesion', 'Left forearm', 'High', 'Under Review', 'Patient reports a growing mole on left forearm, approximately 8mm diameter. Irregular borders noted during initial exam. Family history of melanoma.', 'santos_maria_forearm_001.jpg', 'INS-20485', 'maria.santos@email.com', '555-0101', 1, 'synced', 'EXT-REF-1001', 'Verified', 1, '2026-04-01 09:15:00'),
('James', 'Chen', '1978-11-22', 'Pacific Dermatology Group', 'Dr. Michael Park', 'Facial Analysis', 'Right cheek', 'Medium', 'New', 'New patient consult for asymmetric pigmentation on right cheek. No prior imaging on file. Patient requests baseline documentation.', 'chen_james_face_002.jpg', 'INS-33712', 'jchen78@email.com', '555-0102', NULL, 'not_synced', NULL, NULL, 1, '2026-04-02 10:30:00'),
('Aisha', 'Johnson', '1992-07-08', 'Coastal Dermatology Clinic', 'Dr. Rachel Stevens', 'Scar Review', 'Upper back', 'Low', 'Verified', 'Follow-up imaging for surgical scar from prior excision. Healing appears normal per clinic notes. Document for patient record.', 'johnson_aisha_back_003.jpg', 'INS-44829', 'aisha.j@email.com', '555-0103', 2, 'synced', 'EXT-REF-1003', 'Verified', 1, '2026-04-02 14:45:00'),
('Robert', 'Williams', '1965-01-30', 'Metro Health Partners', 'Dr. David Kim', 'Follow-Up', 'Left shoulder', 'Urgent', 'Escalated', 'URGENT: Patient has rapidly changing lesion on left shoulder. Prior imaging from 3 months ago shows significant growth. Dermatologist requests immediate review and second opinion.', 'williams_robert_shoulder_004.jpg', 'INS-55193', 'rwilliams65@email.com', '555-0104', 1, 'failed', NULL, 'Not Verified', 2, '2026-04-03 08:00:00'),
('Elena', 'Volkov', '1988-05-14', 'Summit Dermatology Associates', 'Dr. Lisa Tran', 'Skin Lesion', 'Right ankle', 'High', 'Awaiting Clinic Response', 'Persistent rash near ankle with possible vascular involvement. Clinic has been asked to clarify medication history before imaging review can proceed.', 'volkov_elena_ankle_005.jpg', 'INS-61047', 'evolkov88@email.com', '555-0105', 2, 'synced', 'EXT-REF-1005', 'Verified', 1, '2026-04-03 11:20:00'),
('David', 'Park', '2001-09-03', 'Valley Imaging Center', 'Dr. Susan Wright', 'Other', 'Chest', 'Medium', 'New', 'Baseline body mapping requested for patient with multiple atypical nevi. First-time imaging patient. No acute concerns at this time.', NULL, 'INS-72356', 'dpark2001@email.com', '555-0106', NULL, 'not_synced', NULL, NULL, 2, '2026-04-04 09:00:00'),
('Sarah', 'Mitchell', '1975-12-19', 'Riverside Medical Center', 'Dr. Angela Cruz', 'Facial Analysis', 'Forehead', 'Low', 'Closed', 'Completed imaging and review for benign keratosis on forehead. Case closed after dermatologist confirmed no further action needed.', 'mitchell_sarah_forehead_007.jpg', 'INS-83491', 'smitchell75@email.com', '555-0107', 1, 'synced', 'EXT-REF-1007', 'Verified', 1, '2026-04-01 13:30:00'),
('Marcus', 'Thompson', '1990-04-25', 'Coastal Dermatology Clinic', 'Dr. Rachel Stevens', 'Skin Lesion', 'Left thigh', 'High', 'Under Review', 'New lesion identified during routine skin check. Dermoscopy images needed. Patient is anxious and requests expedited review.', 'thompson_marcus_thigh_008.jpg', 'INS-94628', 'mthompson90@email.com', '555-0108', 2, 'not_synced', NULL, NULL, 2, '2026-04-05 10:15:00'),
('Lisa', 'Nguyen', '1983-08-11', 'Pacific Dermatology Group', 'Dr. Michael Park', 'Scar Review', 'Abdomen', 'Medium', 'Verified', 'Post-operative scar review from abdominoplasty. Clinic confirms healing within normal parameters. Images archived for longitudinal tracking.', 'nguyen_lisa_abdomen_009.jpg', 'INS-10574', 'lnguyen83@email.com', '555-0109', 1, 'synced', 'EXT-REF-1009', 'Verified', 1, '2026-04-04 15:00:00'),
('John', 'O''Brien', '1970-06-07', 'Metro Health Partners', 'Dr. David Kim', 'Follow-Up', 'Scalp', 'Urgent', 'Escalated', 'URGENT FOLLOW-UP: Previously biopsied scalp lesion. Pathology results pending. Patient reports increased sensitivity and color change at biopsy site.', 'obrien_john_scalp_010.jpg', 'INS-11693', 'jobrien70@email.com', '555-0110', 1, 'failed', NULL, 'Not Verified', 2, '2026-04-05 08:30:00'),
('Priya', 'Sharma', '1995-02-28', 'Summit Dermatology Associates', 'Dr. Lisa Tran', 'Skin Lesion', 'Right wrist', 'Medium', 'New', 'Small raised lesion on dorsal wrist. Patient noticed it 2 weeks ago. No pain or itching. Requesting imaging for documentation and monitoring.', NULL, 'INS-12847', 'psharma95@email.com', '555-0111', NULL, 'not_synced', NULL, NULL, 1, '2026-04-06 11:00:00'),
('Carlos', 'Reyes', '1968-10-16', 'Valley Imaging Center', 'Dr. Susan Wright', 'Facial Analysis', 'Nose bridge', 'High', 'Under Review', 'Actinic keratosis suspected on nose bridge. Patient has history of sun exposure. Clinic requests detailed imaging for treatment planning.', 'reyes_carlos_nose_012.jpg', 'INS-13962', 'creyes68@email.com', '555-0112', 2, 'synced', 'EXT-REF-1012', 'Verified', 2, '2026-04-06 14:30:00');

-- ------------------------------------------------------------
-- Case Status History
-- ------------------------------------------------------------
INSERT INTO case_status_history (case_id, old_status, new_status, changed_by, changed_at, notes) VALUES
(1, 'New', 'Under Review', 1, '2026-04-01 10:00:00', 'Assigned for priority review due to family history.'),
(3, 'New', 'Under Review', 1, '2026-04-02 15:00:00', 'Initial review started.'),
(3, 'Under Review', 'Verified', 2, '2026-04-03 09:00:00', 'Imaging confirmed normal healing. Case verified.'),
(4, 'New', 'Under Review', 2, '2026-04-03 08:30:00', 'Urgent case flagged for immediate attention.'),
(4, 'Under Review', 'Escalated', 1, '2026-04-03 09:15:00', 'Escalated due to rapid lesion changes. Second opinion required.'),
(5, 'New', 'Under Review', 1, '2026-04-03 12:00:00', 'Review initiated.'),
(5, 'Under Review', 'Awaiting Clinic Response', 1, '2026-04-03 14:00:00', 'Medication history needed from clinic before proceeding.'),
(7, 'New', 'Under Review', 1, '2026-04-01 14:00:00', 'Routine review.'),
(7, 'Under Review', 'Verified', 1, '2026-04-02 10:00:00', 'Confirmed benign. No action needed.'),
(7, 'Verified', 'Closed', 1, '2026-04-03 16:00:00', 'Case closed per dermatologist recommendation.'),
(8, 'New', 'Under Review', 2, '2026-04-05 11:00:00', 'Patient requested expedited review.'),
(9, 'New', 'Under Review', 1, '2026-04-04 15:30:00', 'Standard review.'),
(9, 'Under Review', 'Verified', 1, '2026-04-05 10:00:00', 'Healing confirmed normal.'),
(10, 'New', 'Under Review', 2, '2026-04-05 09:00:00', 'Urgent follow-up initiated.'),
(10, 'Under Review', 'Escalated', 1, '2026-04-05 09:45:00', 'Escalated: pathology pending + symptom changes.'),
(12, 'New', 'Under Review', 2, '2026-04-06 15:00:00', 'Detailed imaging review in progress.');

-- ------------------------------------------------------------
-- Support Notes
-- ------------------------------------------------------------
INSERT INTO support_notes (case_id, author_id, note_body, note_type, created_at) VALUES
(4, 1, 'Contacted Metro Health Partners for prior imaging records. Receptionist confirmed records will be faxed within 24 hours.', 'support', '2026-04-03 10:00:00'),
(4, 2, 'REST sync to external system failed with timeout error. Will retry after network maintenance window.', 'sync_issue', '2026-04-03 11:00:00'),
(4, 1, 'Patient called asking about status. Informed them case is under expedited review. Documented patient concern.', 'customer_issue', '2026-04-03 14:00:00'),
(5, 1, 'Left voicemail with Summit Dermatology for medication history. Will follow up tomorrow if no response.', 'support', '2026-04-03 15:00:00'),
(10, 2, 'Pathology lab contacted—results expected within 48 hours. Monitoring case closely.', 'support', '2026-04-05 10:00:00'),
(10, 1, 'SOAP verification failed for this case. Insurance ID may be incorrect. Checking with clinic.', 'technical', '2026-04-05 11:00:00'),
(10, 1, 'Clinic confirmed insurance ID is correct. Retrying SOAP verification.', 'support', '2026-04-05 14:00:00'),
(1, 2, 'Reviewed dermoscopy images. Recommend biopsy based on asymmetry. Notified provider.', 'technical', '2026-04-02 09:00:00'),
(8, 2, 'Patient expressed anxiety during phone call. Assured them of expedited timeline. Noted for care team.', 'customer_issue', '2026-04-05 12:00:00'),
(12, 2, 'High-resolution images received from Valley Imaging. Proceeding with detailed analysis.', 'technical', '2026-04-07 09:00:00');

-- ------------------------------------------------------------
-- Integration Logs (REST sync attempts)
-- ------------------------------------------------------------
INSERT INTO integration_logs (case_id, endpoint, request_payload, response_payload, http_status, success, error_message, attempted_by, attempted_at) VALUES
(1, '/api/rest-external-endpoint.php', '{"case_id":1,"patient_name":"Maria Santos","clinic":"Riverside Medical Center","imaging_type":"Skin Lesion","priority":"High","status":"Under Review"}', '{"success":true,"external_reference_id":"EXT-REF-1001","message":"Case received and registered.","received_at":"2026-04-01 10:30:00"}', 200, 1, NULL, 1, '2026-04-01 10:30:00'),
(3, '/api/rest-external-endpoint.php', '{"case_id":3,"patient_name":"Aisha Johnson","clinic":"Coastal Dermatology Clinic","imaging_type":"Scar Review","priority":"Low","status":"Verified"}', '{"success":true,"external_reference_id":"EXT-REF-1003","message":"Case received and registered.","received_at":"2026-04-03 09:30:00"}', 200, 1, NULL, 1, '2026-04-03 09:30:00'),
(4, '/api/rest-external-endpoint.php', '{"case_id":4,"patient_name":"Robert Williams","clinic":"Metro Health Partners","imaging_type":"Follow-Up","priority":"Urgent","status":"Escalated"}', NULL, 0, 0, 'Connection timed out after 30 seconds. External service unreachable.', 2, '2026-04-03 11:00:00'),
(5, '/api/rest-external-endpoint.php', '{"case_id":5,"patient_name":"Elena Volkov","clinic":"Summit Dermatology Associates","imaging_type":"Skin Lesion","priority":"High","status":"Awaiting Clinic Response"}', '{"success":true,"external_reference_id":"EXT-REF-1005","message":"Case received and registered.","received_at":"2026-04-03 14:30:00"}', 200, 1, NULL, 1, '2026-04-03 14:30:00'),
(7, '/api/rest-external-endpoint.php', '{"case_id":7,"patient_name":"Sarah Mitchell","clinic":"Riverside Medical Center","imaging_type":"Facial Analysis","priority":"Low","status":"Closed"}', '{"success":true,"external_reference_id":"EXT-REF-1007","message":"Case received and registered.","received_at":"2026-04-03 16:30:00"}', 200, 1, NULL, 1, '2026-04-03 16:30:00'),
(9, '/api/rest-external-endpoint.php', '{"case_id":9,"patient_name":"Lisa Nguyen","clinic":"Pacific Dermatology Group","imaging_type":"Scar Review","priority":"Medium","status":"Verified"}', '{"success":true,"external_reference_id":"EXT-REF-1009","message":"Case received and registered.","received_at":"2026-04-05 10:30:00"}', 200, 1, NULL, 1, '2026-04-05 10:30:00'),
(10, '/api/rest-external-endpoint.php', '{"case_id":10,"patient_name":"John O''Brien","clinic":"Metro Health Partners","imaging_type":"Follow-Up","priority":"Urgent","status":"Escalated"}', NULL, 500, 0, 'External service returned internal server error. Payload may be malformed.', 1, '2026-04-05 10:00:00'),
(12, '/api/rest-external-endpoint.php', '{"case_id":12,"patient_name":"Carlos Reyes","clinic":"Valley Imaging Center","imaging_type":"Facial Analysis","priority":"High","status":"Under Review"}', '{"success":true,"external_reference_id":"EXT-REF-1012","message":"Case received and registered.","received_at":"2026-04-07 09:30:00"}', 200, 1, NULL, 2, '2026-04-07 09:30:00');

-- ------------------------------------------------------------
-- SOAP Verifications
-- ------------------------------------------------------------
INSERT INTO soap_verifications (case_id, clinic_name, insurance_id, verification_status, clinic_approved, policy_type, message, checked_at, created_at) VALUES
(1, 'Riverside Medical Center', 'INS-20485', 'Verified', 1, 'Full Coverage', 'Clinic and insurance verified successfully.', '2026-04-01 09:30:00', '2026-04-01 09:30:00'),
(3, 'Coastal Dermatology Clinic', 'INS-44829', 'Verified', 1, 'Full Coverage', 'Clinic and insurance verified successfully.', '2026-04-02 15:00:00', '2026-04-02 15:00:00'),
(4, 'Metro Health Partners', 'INS-55193', 'Not Verified', 0, 'Unknown', 'Unable to verify coverage. Insurance ID format not recognized.', '2026-04-03 08:15:00', '2026-04-03 08:15:00'),
(5, 'Summit Dermatology Associates', 'INS-61047', 'Verified', 1, 'Partial Coverage', 'Clinic verified. Insurance covers diagnostic imaging only.', '2026-04-03 11:30:00', '2026-04-03 11:30:00'),
(7, 'Riverside Medical Center', 'INS-83491', 'Verified', 1, 'Full Coverage', 'Clinic and insurance verified successfully.', '2026-04-01 13:45:00', '2026-04-01 13:45:00'),
(9, 'Pacific Dermatology Group', 'INS-10574', 'Verified', 1, 'Full Coverage', 'Clinic and insurance verified successfully.', '2026-04-04 15:15:00', '2026-04-04 15:15:00'),
(10, 'Metro Health Partners', 'INS-11693', 'Not Verified', 0, 'Unknown', 'Verification service returned an error. Please retry.', '2026-04-05 08:45:00', '2026-04-05 08:45:00'),
(12, 'Valley Imaging Center', 'INS-13962', 'Verified', 1, 'Full Coverage', 'Clinic and insurance verified successfully.', '2026-04-06 14:45:00', '2026-04-06 14:45:00');

-- ------------------------------------------------------------
-- Sample Application Logs
-- ------------------------------------------------------------
INSERT INTO app_logs (level, message, context, file, line, created_at) VALUES
('info', 'Application started successfully.', NULL, 'index.php', 1, '2026-04-01 08:00:00'),
('info', 'User admin logged in.', '{"user_id":1}', 'login.php', 45, '2026-04-01 09:00:00'),
('error', 'REST sync failed for case #4: Connection timed out.', '{"case_id":4,"endpoint":"/api/rest-external-endpoint.php"}', 'IntegrationService.php', 78, '2026-04-03 11:00:00'),
('warning', 'SOAP verification returned Not Verified for case #4.', '{"case_id":4,"insurance_id":"INS-55193"}', 'SoapVerificationService.php', 52, '2026-04-03 08:15:00'),
('error', 'REST sync failed for case #10: HTTP 500 from external service.', '{"case_id":10,"http_status":500}', 'IntegrationService.php', 78, '2026-04-05 10:00:00'),
('warning', 'SOAP verification returned Not Verified for case #10.', '{"case_id":10,"insurance_id":"INS-11693"}', 'SoapVerificationService.php', 52, '2026-04-05 08:45:00'),
('info', 'User sarah logged in.', '{"user_id":2}', 'login.php', 45, '2026-04-05 08:00:00'),
('info', 'New case #11 created by user admin.', '{"case_id":11,"patient":"Priya Sharma"}', 'CaseService.php', 102, '2026-04-06 11:00:00');

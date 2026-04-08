-- ─── Clinical Imaging QA Lab — SQL Validation Queries ───
-- Use these queries to manually verify database state after key workflows.

-- 1. Count all captures by status
SELECT capture_status, COUNT(*) as count
FROM captures
GROUP BY capture_status
ORDER BY count DESC;

-- 2. Verify a specific capture was inserted
-- Replace :patient_id with the actual patient ID
SELECT id, patient_id, session_id, image_type, capture_status, error_message, retry_count, created_at
FROM captures
WHERE patient_id = :patient_id
ORDER BY created_at DESC;

-- 3. Verify failed captures have error messages
SELECT id, patient_id, capture_status, error_message
FROM captures
WHERE capture_status = 'failed' AND error_message IS NULL;
-- Expected: 0 rows (all failed captures should have error messages)

-- 4. Verify retry_count incremented
SELECT id, patient_id, retry_count, capture_status
FROM captures
WHERE retry_count > 0
ORDER BY retry_count DESC;

-- 5. Count defects by severity
SELECT severity, COUNT(*) as count
FROM defects
GROUP BY severity
ORDER BY count DESC;

-- 6. Verify all open defects
SELECT id, title, severity, priority, status, created_at
FROM defects
WHERE status = 'open'
ORDER BY created_at DESC;

-- 7. Dashboard-equivalent summary counts
SELECT
    (SELECT COUNT(*) FROM captures) as total_captures,
    (SELECT COUNT(*) FROM captures WHERE capture_status = 'success') as successful_captures,
    (SELECT COUNT(*) FROM captures WHERE capture_status = 'failed') as failed_captures,
    (SELECT COUNT(*) FROM captures WHERE capture_status = 'pending') as pending_captures,
    (SELECT COUNT(*) FROM defects) as total_defects,
    (SELECT COUNT(*) FROM defects WHERE status = 'open') as open_defects;

-- 8. Recent device events for audit trail
SELECT id, device_name, event_type, details, created_at
FROM device_events
ORDER BY created_at DESC
LIMIT 20;

-- 9. Verify capture and defect totals add up
SELECT
    (SELECT COUNT(*) FROM captures) =
    (SELECT COUNT(*) FROM captures WHERE capture_status IN ('success', 'failed', 'pending'))
    AS captures_status_consistent;

-- 10. Find captures with suspicious data
SELECT id, patient_id, capture_status, file_path
FROM captures
WHERE capture_status = 'success' AND file_path IS NULL;
-- Expected: 0 rows (successful captures should always have a file path)

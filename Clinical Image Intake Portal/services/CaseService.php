<?php
/**
 * CaseService
 * Clinical Image Intake Portal
 *
 * Business logic for managing imaging cases: CRUD operations,
 * status updates, status history, and support notes.
 */

require_once __DIR__ . '/../config/database.php';
require_once __DIR__ . '/../config/config.php';
require_once __DIR__ . '/../includes/functions.php';
require_once __DIR__ . '/LogService.php';

class CaseService
{
    private PDO $pdo;
    private LogService $logger;

    public function __construct()
    {
        $this->pdo = getDbConnection();
        $this->logger = new LogService();
    }

    // ── Case Retrieval ───────────────────────────────────────

    /**
     * Get paginated cases with optional filters.
     *
     * @param array $filters  Associative array of filter criteria
     * @param int   $page     Current page number (1-indexed)
     * @param int   $perPage  Results per page
     * @return array
     */
    public function getAllCases(array $filters = [], int $page = 1, int $perPage = CASES_PER_PAGE): array
    {
        $where = [];
        $params = [];

        // Search filter (patient name or clinic)
        if (!empty($filters['search'])) {
            $search = '%' . $filters['search'] . '%';
            $where[] = '(c.patient_first_name LIKE ? OR c.patient_last_name LIKE ? OR c.clinic_name LIKE ?)';
            $params[] = $search;
            $params[] = $search;
            $params[] = $search;
        }

        // Status filter
        if (!empty($filters['status'])) {
            $where[] = 'c.status = ?';
            $params[] = $filters['status'];
        }

        // Priority filter
        if (!empty($filters['priority'])) {
            $where[] = 'c.priority = ?';
            $params[] = $filters['priority'];
        }

        // Imaging type filter
        if (!empty($filters['imaging_type'])) {
            $where[] = 'c.imaging_type = ?';
            $params[] = $filters['imaging_type'];
        }

        $whereClause = $where ? 'WHERE ' . implode(' AND ', $where) : '';
        $offset = ($page - 1) * $perPage;

        $sql = "SELECT c.*, u.full_name AS assigned_name
                FROM cases c
                LEFT JOIN users u ON c.assigned_to = u.id
                {$whereClause}
                ORDER BY c.created_at DESC
                LIMIT ? OFFSET ?";
        $params[] = $perPage;
        $params[] = $offset;

        $stmt = $this->pdo->prepare($sql);
        $stmt->execute($params);
        return $stmt->fetchAll();
    }

    /**
     * Get total count of cases matching filters (for pagination).
     */
    public function getTotalCasesCount(array $filters = []): int
    {
        $where = [];
        $params = [];

        if (!empty($filters['search'])) {
            $search = '%' . $filters['search'] . '%';
            $where[] = '(patient_first_name LIKE ? OR patient_last_name LIKE ? OR clinic_name LIKE ?)';
            $params[] = $search;
            $params[] = $search;
            $params[] = $search;
        }
        if (!empty($filters['status'])) {
            $where[] = 'status = ?';
            $params[] = $filters['status'];
        }
        if (!empty($filters['priority'])) {
            $where[] = 'priority = ?';
            $params[] = $filters['priority'];
        }
        if (!empty($filters['imaging_type'])) {
            $where[] = 'imaging_type = ?';
            $params[] = $filters['imaging_type'];
        }

        $whereClause = $where ? 'WHERE ' . implode(' AND ', $where) : '';
        $sql = "SELECT COUNT(*) FROM cases {$whereClause}";

        $stmt = $this->pdo->prepare($sql);
        $stmt->execute($params);
        return (int) $stmt->fetchColumn();
    }

    /**
     * Get a single case by ID with assigned user info.
     */
    public function getCaseById(int $id): ?array
    {
        $stmt = $this->pdo->prepare(
            'SELECT c.*, u.full_name AS assigned_name
             FROM cases c
             LEFT JOIN users u ON c.assigned_to = u.id
             WHERE c.id = ?'
        );
        $stmt->execute([$id]);
        $case = $stmt->fetch();
        return $case ?: null;
    }

    // ── Case Creation ────────────────────────────────────────

    /**
     * Create a new imaging case.
     *
     * @param array $data  Validated case data
     * @return int  The new case ID
     */
    public function createCase(array $data): int
    {
        $stmt = $this->pdo->prepare(
            'INSERT INTO cases (
                patient_first_name, patient_last_name, date_of_birth,
                clinic_name, provider_name, imaging_type, body_area,
                priority, status, symptoms_notes, image_filename,
                insurance_id, patient_email, patient_phone,
                assigned_to, created_by
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, "New", ?, ?, ?, ?, ?, ?, ?)'
        );

        $stmt->execute([
            $data['patient_first_name'],
            $data['patient_last_name'],
            $data['date_of_birth'],
            $data['clinic_name'],
            $data['provider_name'],
            $data['imaging_type'],
            $data['body_area'] ?? null,
            $data['priority'],
            $data['symptoms_notes'] ?? null,
            $data['image_filename'] ?? null,
            $data['insurance_id'] ?? null,
            $data['patient_email'] ?? null,
            $data['patient_phone'] ?? null,
            $data['assigned_to'] ?: null,
            $data['created_by'],
        ]);

        $caseId = (int) $this->pdo->lastInsertId();

        // Record initial status in history
        $this->recordStatusChange($caseId, null, 'New', $data['created_by'], 'Case created.');

        $this->logger->log('info', "New case #{$caseId} created.", [
            'case_id' => $caseId,
            'patient' => $data['patient_first_name'] . ' ' . $data['patient_last_name'],
        ], 'CaseService.php', __LINE__);

        return $caseId;
    }

    // ── Status Updates ───────────────────────────────────────

    /**
     * Update the status of a case and record the change.
     */
    public function updateCaseStatus(int $caseId, string $newStatus, int $userId, ?string $notes = null): bool
    {
        // Validate status
        if (!in_array($newStatus, getStatusOptions())) {
            return false;
        }

        // Get current status
        $current = $this->getCaseById($caseId);
        if (!$current) return false;

        $oldStatus = $current['status'];

        // Update case
        $stmt = $this->pdo->prepare('UPDATE cases SET status = ? WHERE id = ?');
        $stmt->execute([$newStatus, $caseId]);

        // Record in history
        $this->recordStatusChange($caseId, $oldStatus, $newStatus, $userId, $notes);

        $this->logger->log('info', "Case #{$caseId} status changed from '{$oldStatus}' to '{$newStatus}'.", [
            'case_id' => $caseId,
            'user_id' => $userId,
        ], 'CaseService.php', __LINE__);

        return true;
    }

    /**
     * Record a status change in the history table.
     */
    private function recordStatusChange(int $caseId, ?string $oldStatus, string $newStatus, int $userId, ?string $notes = null): void
    {
        $stmt = $this->pdo->prepare(
            'INSERT INTO case_status_history (case_id, old_status, new_status, changed_by, notes) VALUES (?, ?, ?, ?, ?)'
        );
        $stmt->execute([$caseId, $oldStatus, $newStatus, $userId, $notes]);
    }

    /**
     * Get status change history for a case.
     */
    public function getCaseStatusHistory(int $caseId): array
    {
        $stmt = $this->pdo->prepare(
            'SELECT csh.*, u.full_name AS changed_by_name
             FROM case_status_history csh
             LEFT JOIN users u ON csh.changed_by = u.id
             WHERE csh.case_id = ?
             ORDER BY csh.changed_at DESC'
        );
        $stmt->execute([$caseId]);
        return $stmt->fetchAll();
    }

    // ── Support Notes ────────────────────────────────────────

    /**
     * Add a support/troubleshooting note to a case.
     */
    public function addSupportNote(int $caseId, int $authorId, string $noteBody, string $noteType): int
    {
        if (!in_array($noteType, getNoteTypes())) {
            $noteType = 'support';
        }

        $stmt = $this->pdo->prepare(
            'INSERT INTO support_notes (case_id, author_id, note_body, note_type) VALUES (?, ?, ?, ?)'
        );
        $stmt->execute([$caseId, $authorId, $noteBody, $noteType]);

        return (int) $this->pdo->lastInsertId();
    }

    /**
     * Get all support notes for a case (chronological, newest first).
     */
    public function getSupportNotes(int $caseId): array
    {
        $stmt = $this->pdo->prepare(
            'SELECT sn.*, u.full_name AS author_name
             FROM support_notes sn
             LEFT JOIN users u ON sn.author_id = u.id
             WHERE sn.case_id = ?
             ORDER BY sn.created_at DESC'
        );
        $stmt->execute([$caseId]);
        return $stmt->fetchAll();
    }

    // ── Reporting Queries ────────────────────────────────────

    /**
     * Get count of cases grouped by status.
     */
    public function getCountByStatus(): array
    {
        $stmt = $this->pdo->query(
            'SELECT status, COUNT(*) as count FROM cases GROUP BY status ORDER BY FIELD(status, "New", "Under Review", "Awaiting Clinic Response", "Verified", "Escalated", "Closed")'
        );
        return $stmt->fetchAll();
    }

    /**
     * Get count of cases grouped by imaging type.
     */
    public function getCountByImagingType(): array
    {
        $stmt = $this->pdo->query(
            'SELECT imaging_type, COUNT(*) as count FROM cases GROUP BY imaging_type ORDER BY count DESC'
        );
        return $stmt->fetchAll();
    }

    /**
     * Get count of cases grouped by priority.
     */
    public function getCountByPriority(): array
    {
        $stmt = $this->pdo->query(
            'SELECT priority, COUNT(*) as count FROM cases GROUP BY priority ORDER BY FIELD(priority, "Urgent", "High", "Medium", "Low")'
        );
        return $stmt->fetchAll();
    }

    /**
     * Get total count of open (non-closed) cases.
     */
    public function getOpenCasesCount(): int
    {
        $stmt = $this->pdo->query('SELECT COUNT(*) FROM cases WHERE status != "Closed"');
        return (int) $stmt->fetchColumn();
    }

    /**
     * Get recent escalated cases.
     */
    public function getRecentEscalations(int $limit = 5): array
    {
        $stmt = $this->pdo->prepare(
            'SELECT c.*, u.full_name AS assigned_name
             FROM cases c
             LEFT JOIN users u ON c.assigned_to = u.id
             WHERE c.status = "Escalated"
             ORDER BY c.updated_at DESC
             LIMIT ?'
        );
        $stmt->execute([$limit]);
        return $stmt->fetchAll();
    }

    /**
     * Update the external sync status on a case.
     */
    public function updateSyncStatus(int $caseId, string $syncStatus, ?string $referenceId = null): void
    {
        $stmt = $this->pdo->prepare(
            'UPDATE cases SET external_sync_status = ?, external_reference_id = ? WHERE id = ?'
        );
        $stmt->execute([$syncStatus, $referenceId, $caseId]);
    }

    /**
     * Update the SOAP verification status on a case.
     */
    public function updateSoapStatus(int $caseId, string $verificationStatus): void
    {
        $stmt = $this->pdo->prepare(
            'UPDATE cases SET soap_verification_status = ? WHERE id = ?'
        );
        $stmt->execute([$verificationStatus, $caseId]);
    }
}

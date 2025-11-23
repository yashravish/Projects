# Operations & Support Playbook

Guide for Application Support Engineers managing the Operations Hub.

## 1. Monitoring Integration Health
*   **Where:** Navigate to `/Support/Jobs`.
*   **What to look for:**
    *   Jobs with Status `Failed` (highlighted in red).
    *   Jobs stuck in `Running` for > 1 hour.
*   **Action:** Click "Details" to view error logs. Common errors:
    *   *Connection Refused:* Check if ERP API is down.
    *   *Unknown SKU:* Order contains a product not yet synced. Run Product Sync first.

## 2. Replaying Failed Jobs
If a sync job failed due to a transient issue (e.g., network blip):
1.  Go to `/Support/Jobs`.
2.  Find the failed job.
3.  Click the **Replay** button.
4.  Refresh to see the new `_REPLAY` job start.

## 3. Replenishment & Alerts
*   **Setting Thresholds:** Use the API or future UI to POST to `/api/config/reorder-thresholds`.
*   **Viewing Suggestions:** API endpoint `/api/replenishment-suggestions` lists items needing reorder.
*   **Action:** Warehouse managers should review this list daily.

## 4. Reporting Issues
*   **Fill Rate is 0%:** Ensure orders are being **Allocated** and **Shipped** via the API. Pending orders do not count towards shipped totals.
*   **Missing Inventory:** Check `InventoryMovements` table for audit trail.

## 5. Configuration
*   **Sync Interval:** Modify `appsettings.json` -> `Integration:SyncIntervalMinutes`.
*   **ERP URL:** Modify `appsettings.json` -> `Integration:ErpBaseUrl`.


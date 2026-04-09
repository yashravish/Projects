/**
 * case-detail.js — Case Detail Page JavaScript
 * Clinical Image Intake Portal
 *
 * Handles AJAX interactions on the case detail page:
 * - Status updates
 * - Support note submission
 * - REST sync triggering
 * - SOAP verification triggering
 */

(function ($) {
    'use strict';

    // ── Status Update (Detail Page) ──────────────────────────
    $('#btnUpdateStatus').on('click', function () {
        var $btn = $(this);
        var caseId = $btn.data('case-id');
        var newStatus = $('#detailNewStatus').val();
        var notes = $('#detailStatusNotes').val();

        $btn.prop('disabled', true).text('Updating...');

        ajaxPost('api/update-status.php', {
            case_id: caseId,
            new_status: newStatus,
            notes: notes
        }, function (response) {
            if (response.success) {
                // Update the status badge on the page
                $('#caseStatusBadge')
                    .text(response.new_status)
                    .attr('class', 'badge ' + response.badge_class);

                $('#detailStatusNotes').val('');
                showToast(response.message, 'success');

                // Reload to refresh status history
                setTimeout(function () {
                    location.reload();
                }, 1200);
            } else {
                showToast(response.message, 'error');
            }
            $btn.prop('disabled', false).text('Update Status');
        }, function (msg) {
            showToast(msg, 'error');
            $btn.prop('disabled', false).text('Update Status');
        });
    });

    // ── Add Support Note ─────────────────────────────────────
    $('#btnAddNote').on('click', function () {
        var $btn = $(this);
        var caseId = $btn.data('case-id');
        var noteBody = $('#noteBody').val().trim();
        var noteType = $('#noteType').val();

        if (!noteBody) {
            showToast('Please enter a note.', 'error');
            return;
        }

        $btn.prop('disabled', true).text('Adding...');

        ajaxPost('api/add-note.php', {
            case_id: caseId,
            note_body: noteBody,
            note_type: noteType
        }, function (response) {
            if (response.success) {
                var note = response.note;

                // Remove "no notes" message if present
                $('#noNotesMsg').remove();

                // Prepend the new note to the list
                var noteHtml = '<div class="note-item" style="animation: toastIn 0.3s ease;">' +
                    '<div class="note-header">' +
                        '<span class="note-author">' + escapeHtml(note.author_name) + '</span>' +
                        '<span class="note-type-badge badge-' + escapeHtml(note.note_type) + '">' + escapeHtml(note.type_label) + '</span>' +
                        '<span class="note-date">' + escapeHtml(note.created_at) + '</span>' +
                    '</div>' +
                    '<div class="note-body">' + escapeHtml(note.note_body) + '</div>' +
                '</div>';

                $('#notesList').prepend(noteHtml);
                $('#noteBody').val('');
                showToast(response.message, 'success');
            } else {
                showToast(response.message, 'error');
            }
            $btn.prop('disabled', false).text('Add Note');
        }, function (msg) {
            showToast(msg, 'error');
            $btn.prop('disabled', false).text('Add Note');
        });
    });

    // ── REST Sync Case ───────────────────────────────────────
    $('#btnSyncCase').on('click', function () {
        var $btn = $(this);
        var caseId = $btn.data('case-id');

        $btn.prop('disabled', true).text('Syncing...');

        ajaxPost('api/sync-case.php', {
            case_id: caseId
        }, function (response) {
            if (response.success) {
                showToast(response.message, 'success');
                // Update sync badge
                $('#syncStatusBadge')
                    .text('Synced')
                    .attr('class', 'badge badge-verified');
                $btn.text('⇄ Sync Now');

                // Reload to show updated integration logs
                setTimeout(function () {
                    location.reload();
                }, 1500);
            } else {
                showToast(response.message || 'Sync failed.', 'error');
                $('#syncStatusBadge')
                    .text('Failed')
                    .attr('class', 'badge badge-escalated');
                $btn.text('↻ Retry Sync');
            }
            $btn.prop('disabled', false);
        }, function (msg) {
            showToast(msg, 'error');
            $btn.prop('disabled', false).text('↻ Retry Sync');
        });
    });

    // ── SOAP Verification ────────────────────────────────────
    $('#btnVerifySoap').on('click', function () {
        var $btn = $(this);
        var caseId = $btn.data('case-id');
        var clinic = $btn.data('clinic');
        var insurance = $btn.data('insurance');

        if (!insurance) {
            showToast('No insurance ID on file. Cannot verify coverage.', 'error');
            return;
        }

        $btn.prop('disabled', true).text('Verifying...');

        ajaxPost('api/verify-soap.php', {
            case_id: caseId,
            clinic_name: clinic,
            insurance_id: insurance
        }, function (response) {
            if (response.success) {
                showToast('Verification complete: ' + response.data.verification_status, 'success');
                // Reload to show updated verification result
                setTimeout(function () {
                    location.reload();
                }, 1500);
            } else {
                showToast(response.message || 'Verification failed.', 'error');
            }
            $btn.prop('disabled', false).text('↻ Re-Verify');
        }, function (msg) {
            showToast(msg, 'error');
            $btn.prop('disabled', false).text('↻ Re-Verify');
        });
    });

})(jQuery);

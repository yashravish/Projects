/**
 * dashboard.js — Dashboard Page JavaScript
 * Clinical Image Intake Portal
 *
 * Handles the status update modal and AJAX status changes
 * from the case dashboard table.
 */

(function ($) {
    'use strict';

    var currentCaseId = null;

    // ── Open Status Modal ────────────────────────────────────
    $(document).on('click', '.status-update-btn', function () {
        currentCaseId = $(this).data('case-id');
        var currentStatus = $(this).data('current-status');

        $('#modalCaseId').val(currentCaseId);
        $('#modalCaseLabel').text('#' + currentCaseId);
        $('#modalNewStatus').val(currentStatus);
        $('#modalStatusNotes').val('');
        $('#statusModal').fadeIn(150);
    });

    // ── Close Status Modal ───────────────────────────────────
    function closeModal() {
        $('#statusModal').fadeOut(150);
        currentCaseId = null;
    }

    $('#statusModalClose, #statusModalCancel').on('click', closeModal);

    // Close on overlay click
    $('#statusModal').on('click', function (e) {
        if ($(e.target).is('.modal-overlay')) {
            closeModal();
        }
    });

    // Close on Escape key
    $(document).on('keydown', function (e) {
        if (e.key === 'Escape' && $('#statusModal').is(':visible')) {
            closeModal();
        }
    });

    // ── Submit Status Update ─────────────────────────────────
    $('#statusModalSubmit').on('click', function () {
        var caseId = $('#modalCaseId').val();
        var newStatus = $('#modalNewStatus').val();
        var notes = $('#modalStatusNotes').val();

        if (!caseId || !newStatus) {
            showToast('Please select a status.', 'error');
            return;
        }

        // Disable button during request
        var $btn = $(this);
        $btn.prop('disabled', true).text('Updating...');

        ajaxPost('api/update-status.php', {
            case_id: parseInt(caseId),
            new_status: newStatus,
            notes: notes
        }, function (response) {
            if (response.success) {
                // Update the badge in the table
                var $badge = $('#statusBadge-' + caseId);
                $badge.text(response.new_status)
                      .attr('class', 'badge ' + response.badge_class);

                // Update the button's data attribute
                $('.status-update-btn[data-case-id="' + caseId + '"]')
                    .data('current-status', response.new_status);

                // If status is Closed, hide the status button
                if (response.new_status === 'Closed') {
                    $('.status-update-btn[data-case-id="' + caseId + '"]').remove();
                }

                showToast(response.message, 'success');
                closeModal();
            } else {
                showToast(response.message || 'Update failed.', 'error');
            }
            $btn.prop('disabled', false).text('Update Status');
        }, function (msg) {
            showToast(msg, 'error');
            $btn.prop('disabled', false).text('Update Status');
        });
    });

})(jQuery);

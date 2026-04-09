/**
 * app.js — Shared Application JavaScript
 * Clinical Image Intake Portal
 *
 * Toast notifications, AJAX helpers, and global utilities.
 */

(function ($) {
    'use strict';

    // ── Toast Notification System ────────────────────────────
    window.showToast = function (message, type) {
        type = type || 'info';
        var $container = $('#toastContainer');
        var $toast = $('<div class="toast toast-' + type + '">' +
            '<span>' + escapeHtml(message) + '</span>' +
            '</div>');

        $container.append($toast);

        // Auto-remove after 4 seconds
        setTimeout(function () {
            $toast.css('animation', 'toastOut 0.3s ease forwards');
            setTimeout(function () {
                $toast.remove();
            }, 300);
        }, 4000);
    };

    // ── HTML Escape Utility ──────────────────────────────────
    window.escapeHtml = function (text) {
        if (!text) return '';
        var map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return String(text).replace(/[&<>"']/g, function (m) { return map[m]; });
    };

    // ── AJAX Helper with CSRF ────────────────────────────────
    window.ajaxPost = function (url, data, successCallback, errorCallback) {
        $.ajax({
            url: APP.baseUrl + '/' + url.replace(/^\//, ''),
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(data),
            headers: {
                'X-CSRF-Token': APP.csrfToken,
                'X-Requested-With': 'XMLHttpRequest'
            },
            dataType: 'json',
            success: function (response) {
                if (typeof successCallback === 'function') {
                    successCallback(response);
                }
            },
            error: function (xhr) {
                var msg = 'An error occurred. Please try again.';
                try {
                    var resp = JSON.parse(xhr.responseText);
                    if (resp.message) msg = resp.message;
                } catch (e) { /* ignore parse errors */ }

                if (typeof errorCallback === 'function') {
                    errorCallback(msg, xhr);
                } else {
                    showToast(msg, 'error');
                }
            }
        });
    };

    // ── Auto-dismiss flash alerts ────────────────────────────
    $(function () {
        var $flash = $('#flashAlert');
        if ($flash.length) {
            setTimeout(function () {
                $flash.fadeOut(400, function () {
                    $flash.remove();
                });
            }, 5000);
        }
    });

})(jQuery);

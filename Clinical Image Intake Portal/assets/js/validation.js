/**
 * validation.js — Client-Side Form Validation
 * Clinical Image Intake Portal
 *
 * jQuery-based validation for the New Case Intake form.
 * Validates required fields, email format, phone format,
 * and date validity before allowing form submission.
 */

(function ($) {
    'use strict';

    $('#intakeForm').on('submit', function (e) {
        var isValid = true;

        // Clear all previous errors
        $('.invalid-feedback').text('');
        $('.form-control').removeClass('is-invalid');

        // ── Required Field Checks ────────────────────────────
        var requiredFields = [
            { id: 'patient_first_name', label: 'Patient first name' },
            { id: 'patient_last_name',  label: 'Patient last name' },
            { id: 'date_of_birth',      label: 'Date of birth' },
            { id: 'clinic_name',        label: 'Clinic name' },
            { id: 'provider_name',      label: 'Provider name' }
        ];

        requiredFields.forEach(function (field) {
            var $input = $('#' + field.id);
            if (!$input.val() || !$input.val().trim()) {
                setError(field.id, field.label + ' is required.');
                isValid = false;
            }
        });

        // ── Imaging Type ─────────────────────────────────────
        var $imagingType = $('#imaging_type');
        if (!$imagingType.val()) {
            setError('imaging_type', 'Please select an imaging type.');
            isValid = false;
        }

        // ── Date of Birth Validation ─────────────────────────
        var dob = $('#date_of_birth').val();
        if (dob) {
            var dobDate = new Date(dob);
            var today = new Date();
            if (isNaN(dobDate.getTime())) {
                setError('date_of_birth', 'Invalid date format.');
                isValid = false;
            } else if (dobDate > today) {
                setError('date_of_birth', 'Date of birth cannot be in the future.');
                isValid = false;
            }
        }

        // ── Email Validation (optional field) ────────────────
        var email = $('#patient_email').val().trim();
        if (email) {
            var emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
            if (!emailRegex.test(email)) {
                setError('patient_email', 'Invalid email address format.');
                isValid = false;
            }
        }

        // ── Phone Validation (optional field) ────────────────
        var phone = $('#patient_phone').val().trim();
        if (phone) {
            var phoneRegex = /^[\d\s\-\(\)\+\.]{7,20}$/;
            if (!phoneRegex.test(phone)) {
                setError('patient_phone', 'Invalid phone number format.');
                isValid = false;
            }
        }

        // ── Prevent submission if invalid ────────────────────
        if (!isValid) {
            e.preventDefault();

            // Scroll to first error
            var $firstError = $('.is-invalid').first();
            if ($firstError.length) {
                $('html, body').animate({
                    scrollTop: $firstError.offset().top - 120
                }, 300);
            }
        }
    });

    /**
     * Set an error message and mark the field as invalid.
     */
    function setError(fieldId, message) {
        var $input = $('#' + fieldId);
        $input.addClass('is-invalid');
        $('#err_' + fieldId).text(message);
    }

    // ── Clear error on input change ──────────────────────────
    $('#intakeForm').on('input change', '.form-control', function () {
        var $this = $(this);
        if ($this.hasClass('is-invalid')) {
            $this.removeClass('is-invalid');
            var fieldId = $this.attr('id');
            $('#err_' + fieldId).text('');
        }
    });

})(jQuery);

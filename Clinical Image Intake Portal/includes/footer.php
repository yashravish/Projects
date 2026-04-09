<?php
/**
 * Footer Partial
 * Clinical Image Intake Portal
 *
 * Closes the main layout and includes shared JavaScript files.
 */
?>
        </main><!-- /.main-content -->
    </div><!-- /.app-container -->

    <!-- Toast Container for AJAX notifications -->
    <div class="toast-container" id="toastContainer"></div>

    <footer class="app-footer">
        <p>&copy; <?= date('Y') ?> <?= h(APP_NAME) ?> &mdash; v<?= APP_VERSION ?> &mdash; Internal Use Only</p>
    </footer>

    <!-- jQuery CDN -->
    <script src="https://code.jquery.com/jquery-3.7.1.min.js"
            integrity="sha256-/JqT3SQfawRcv/BIHPThkBvs0OEvtFFmqPF/lYI/Cxo="
            crossorigin="anonymous"></script>

    <!-- Application JavaScript -->
    <script src="<?= BASE_URL ?>assets/js/app.js"></script>

    <?php if (isset($pageScripts) && is_array($pageScripts)): ?>
        <?php foreach ($pageScripts as $script): ?>
            <script src="<?= BASE_URL ?>assets/js/<?= h($script) ?>"></script>
        <?php endforeach; ?>
    <?php endif; ?>
</body>
</html>

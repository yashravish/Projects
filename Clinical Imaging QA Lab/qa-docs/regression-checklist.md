# Regression Checklist — Clinical Imaging QA Lab

Use this checklist before each release or after significant code changes.

## Dashboard
- [ ] Dashboard loads without errors
- [ ] Device status indicator shows correct state (online/offline)
- [ ] Summary stat cards display correct counts
- [ ] Recent captures table shows latest 5 captures
- [ ] Recent defects table shows latest 5 defects
- [ ] All navigation links work correctly
- [ ] Dashboard data matches database state

## Capture Workflow
- [ ] Capture form loads with all fields visible
- [ ] Patient ID, Session ID, and Image Type are required
- [ ] Empty form submission shows validation errors
- [ ] Valid submission triggers device capture
- [ ] Successful capture shows success alert with file path
- [ ] Failed capture shows error alert with message
- [ ] Retry button appears on failed captures
- [ ] Retry button triggers new capture attempt
- [ ] Capture result stored in database with correct status

## History Page
- [ ] History table loads with all capture records
- [ ] Table shows ID, Patient, Session, Type, Status, Device, Retries, Date
- [ ] Status badges display with correct colors
- [ ] Retry button appears for failed captures only
- [ ] Retry action works and table refreshes
- [ ] Refresh button reloads table data
- [ ] Table is scrollable on mobile viewports

## Defect Tracker
- [ ] Defect form loads with all fields
- [ ] Title, Severity, and Priority are required
- [ ] Empty form submission shows validation errors
- [ ] Valid submission shows success message
- [ ] Form clears after successful submission
- [ ] Defects table shows newest defects first
- [ ] Severity and priority badges show correct colors
- [ ] Optional fields (environment, steps, expected, actual) are accepted

## Device Simulator
- [ ] Device status returns online by default
- [ ] Disconnect sets device to offline
- [ ] Reconnect brings device back online
- [ ] Capture succeeds when device is online
- [ ] Capture fails when device is offline
- [ ] Failure modes (timeout, random, corrupted, unavailable) work correctly
- [ ] Reset returns device to default state

## API Health
- [ ] GET /api/health returns 200
- [ ] GET /api/device/status returns valid JSON
- [ ] GET /api/dashboard/summary returns all expected fields
- [ ] POST /api/captures validates input
- [ ] POST /api/defects validates input
- [ ] 404 returned for nonexistent resources

## Cross-Browser
- [ ] All pages load in Chrome/Chromium
- [ ] All pages load in Firefox
- [ ] Forms are functional in both browsers

## Accessibility
- [ ] Skip navigation link present on all pages
- [ ] All form inputs have associated labels
- [ ] Visible focus indicators on interactive elements
- [ ] No critical axe-core violations
- [ ] Semantic heading hierarchy (h1 → h2)
- [ ] ARIA landmarks present (banner, main, contentinfo)

## Performance
- [ ] Dashboard loads within 2 seconds
- [ ] Capture request completes within 5 seconds
- [ ] History page renders within 2 seconds
- [ ] No JavaScript console errors during normal use

# Clinical Image Intake Portal

> **Internal Case Management for Imaging Review Teams**

A production-style PHP web application built for a fictional medical imaging company. The portal enables clinic staff to submit patient imaging case intake forms, manage case workflows, track escalations, and integrate with external systems through REST and SOAP web services.

---

## Why This Project Fits PHP Web Developer Roles

This project directly demonstrates competencies required for entry-level PHP web developer positions at clinical software, healthcare IT, and medical imaging companies:

| Skill Area | How It's Demonstrated |
|---|---|
| **PHP Web Development** | Server-rendered pages, session management, form processing, PDO database access |
| **HTML, CSS, JavaScript, jQuery, AJAX** | Custom responsive UI, client-side validation, asynchronous status updates and note submission |
| **JSON & API Integration** | RESTful endpoint for case sync, JSON request/response handling |
| **REST Web Services** | Mock external REST endpoint with cURL integration, payload validation, error handling |
| **SOAP Web Services** | PHP SoapServer/SoapClient for insurance verification |
| **MySQL Database** | Relational schema with 7 tables, foreign keys, indexes, prepared statements |
| **Translating Wireframes to Pages** | Intake form built from field specifications, dashboard from column requirements |
| **Maintaining Existing Codebase** | Reusable partials (header, footer, sidebar), organized folder structure, service classes |
| **Troubleshooting & Support Tools** | Admin issues page, application logging, integration error tracking |
| **Technical Documentation** | Comprehensive README, inline code comments, API documentation |

---

## Features

### Core Workflow
- **Session-based authentication** with role-based access (Admin, Support Specialist)
- **Case intake form** with 14+ fields, client-side and server-side validation
- **Searchable dashboard** with filters (status, priority, imaging type, patient/clinic search)
- **Pagination** for large case lists
- **Case detail view** with full patient info, symptoms, and clinical metadata

### AJAX-Powered Interactions
- **Status updates** from dashboard (modal) and detail page without full reload
- **Support note submission** with instant DOM insertion
- **REST sync trigger** with real-time feedback
- **SOAP verification trigger** with result display

### Integrations
- **REST API** — Mock external endpoint receives case summaries as JSON, returns reference IDs
- **SOAP Service** — PHP SoapClient calls a local SoapServer for clinic/insurance verification
- **Integration logging** — Every sync attempt is recorded with request/response data

### Admin & Reporting
- **Reports page** — Case counts by status, priority, imaging type; visual bar indicators
- **Issues & Logs page** — Application error/warning viewer with level filtering (Admin only)
- **Escalation tracking** — Recent escalated cases highlighted in reports

### Security
- CSRF token protection on all forms and AJAX requests
- PDO prepared statements for all database queries (SQL injection prevention)
- Output escaping with `htmlspecialchars()` (XSS prevention)
- Secure password hashing with `password_hash()` / `password_verify()`
- Session hardening (httponly cookies, SameSite, regeneration on login)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | PHP 8+ |
| Database | MySQL 5.7+ / MariaDB |
| Frontend | HTML5, CSS3, JavaScript (ES5+), jQuery 3.7 |
| Web Services | REST (cURL + JSON), SOAP (PHP SoapClient/SoapServer) |
| Server | Apache (XAMPP/MAMP) or PHP built-in server |
| Typography | Inter (Google Fonts) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Browser (Client)                      │
│  HTML/CSS/JS/jQuery ─── AJAX ──► JSON API Endpoints     │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────┐
│                 PHP Application Server                   │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────┐    │
│  │  Pages   │  │ Includes │  │     Services        │    │
│  │(Views/   │◄─┤(Partials,│  │ (Business Logic,   │    │
│  │Controllers│  │ Auth,    │  │  DB Queries,       │    │
│  │          │  │ Helpers)  │  │  Integrations)     │    │
│  └──────────┘  └──────────┘  └────────┬───────────┘    │
│                                        │                 │
│  ┌──────────┐  ┌──────────┐  ┌────────┴───────────┐    │
│  │   API    │  │  Config  │  │     Database        │    │
│  │Endpoints │  │          │  │    (MySQL/PDO)      │    │
│  └──────────┘  └──────────┘  └────────────────────┘    │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │           External Service Mocks                  │   │
│  │   REST Endpoint        SOAP Server                │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## Folder Structure

```
clinical-image-intake-portal/
├── api/
│   ├── add-note.php             # AJAX: Add support note
│   ├── rest-external-endpoint.php  # Mock REST external system
│   ├── soap-server.php          # Mock SOAP verification server
│   ├── sync-case.php            # AJAX: Trigger REST sync
│   ├── update-status.php        # AJAX: Update case status
│   └── verify-soap.php          # AJAX: Trigger SOAP verification
├── assets/
│   ├── css/
│   │   └── styles.css           # Complete application styles
│   └── js/
│       ├── app.js               # Shared: Toasts, AJAX helpers
│       ├── case-detail.js       # Case detail page interactions
│       ├── dashboard.js         # Dashboard status modal
│       └── validation.js        # Intake form validation
├── config/
│   ├── config.php               # Application configuration
│   └── database.php             # PDO connection singleton
├── includes/
│   ├── auth.php                 # Session & authentication
│   ├── csrf.php                 # CSRF token management
│   ├── footer.php               # HTML footer partial
│   ├── functions.php            # Helper functions
│   ├── header.php               # HTML header partial
│   └── sidebar.php              # Sidebar navigation partial
├── logs/
│   └── .gitkeep
├── pages/
│   ├── case-detail.php          # Case detail view
│   ├── dashboard.php            # Main case dashboard
│   ├── issues.php               # Admin: Application logs viewer
│   ├── new-case.php             # New case intake form
│   └── reports.php              # Reporting dashboard
├── services/
│   ├── CaseService.php          # Case CRUD & business logic
│   ├── IntegrationService.php   # REST sync service
│   ├── LogService.php           # Application logging
│   └── SoapVerificationService.php  # SOAP client service
├── sql/
│   ├── schema.sql               # Database DDL
│   └── seed.sql                 # Sample data (reference)
├── index.php                    # Entry point / redirect
├── login.php                    # Authentication page
├── logout.php                   # Session teardown
├── setup.php                    # Database setup & seeding
└── README.md                    # This file
```

---

## Database Setup

### Prerequisites
- PHP 8.0+
- MySQL 5.7+ or MariaDB 10.3+
- PHP extensions: `pdo`, `pdo_mysql`, `soap` (optional, for SOAP features)

### Automatic Setup

1. **Configure database credentials** in `config/config.php`:
   ```php
   define('DB_HOST', '127.0.0.1');
   define('DB_PORT', '3306');
   define('DB_NAME', 'clinical_intake_portal');
   define('DB_USER', 'root');
   define('DB_PASS', '');  // Your MySQL password
   ```

2. **Run setup** (CLI or browser):
   ```bash
   php setup.php
   ```
   This creates the database, all tables, and seeds sample data including users with hashed passwords.

### Manual Setup (Alternative)

```bash
mysql -u root -p < sql/schema.sql
mysql -u root -p clinical_intake_portal < sql/seed.sql
```
> **Note:** If using manual setup, user passwords in `seed.sql` are comments only. Use `setup.php` for proper password hashing.

### Database Schema

| Table | Purpose |
|---|---|
| `users` | Login credentials and role assignments |
| `cases` | Core imaging case intake records |
| `case_status_history` | Audit trail of every status change |
| `support_notes` | Troubleshooting & escalation notes |
| `integration_logs` | REST sync attempt records |
| `soap_verifications` | SOAP verification results |
| `app_logs` | System-level error & event logs |

---

## How to Run

### Option A: PHP Built-in Server (Recommended for quick start)

```bash
# Navigate to the project directory
cd clinical-image-intake-portal

# Start the main application server
php -S 127.0.0.1:8000 -t .

# (Optional) In a second terminal, start the SOAP server
php -S 127.0.0.1:8001 -t .
```

Open **http://127.0.0.1:8000** in your browser.

### Option B: XAMPP / MAMP / LAMP

1. Copy the project folder to your web server's document root (e.g., `htdocs/`)
2. Update `BASE_URL` in `config/config.php` if in a subdirectory:
   ```php
   define('BASE_URL', '/clinical-image-intake-portal/');
   ```
3. Start Apache and MySQL
4. Navigate to **http://localhost/clinical-image-intake-portal/**

### First Run

1. Open `http://127.0.0.1:8000/setup.php` to initialize the database
2. Login with demo credentials (below)

---

## Demo Credentials

| Username | Password | Role |
|---|---|---|
| `admin` | `admin123` | Admin |
| `sarah` | `support123` | Support Specialist |

**Admin** has full access including the Issues & Logs page.
**Support Specialist** can manage cases, add notes, and trigger integrations.

---

## Demo Workflow

1. **Log in** with `admin` / `admin123`
2. **View dashboard** — See all 12 sample cases with status badges
3. **Filter cases** — Try filtering by status "Escalated" or priority "Urgent"
4. **Create a new case** — Click "New Case" and fill out the intake form
5. **View case details** — Click any case to see full info, history, and notes
6. **Update status via AJAX** — Change status from the detail page or dashboard modal
7. **Add a support note** — Write a troubleshooting note on a case detail page
8. **Trigger REST sync** — Click "Sync Now" on a case detail page
9. **Trigger SOAP verification** — Click "Verify Coverage" on a case with an insurance ID
10. **Review reports** — See case metrics and escalation summary
11. **View admin issues** — Check application logs and error history

---

## REST Integration

### Endpoint
`POST /api/rest-external-endpoint.php`

### Request (JSON)
```json
{
  "case_id": 1,
  "patient_name": "Maria Santos",
  "date_of_birth": "1985-03-15",
  "clinic": "Riverside Medical Center",
  "provider": "Dr. Angela Cruz",
  "imaging_type": "Skin Lesion",
  "priority": "High",
  "status": "Under Review",
  "insurance_id": "INS-20485"
}
```

### Response (Success — 200)
```json
{
  "success": true,
  "external_reference_id": "EXT-REF-0001",
  "message": "Case received and registered.",
  "received_at": "2026-04-09 12:00:00"
}
```

### Response (Error — 400)
```json
{
  "success": false,
  "message": "Missing required fields: patient_name, clinic",
  "received_at": "2026-04-09 12:00:00"
}
```

### Integration Flow
1. User clicks "Sync Now" on case detail page
2. `IntegrationService` builds JSON payload from case data
3. Sends POST via cURL to the mock endpoint
4. Logs the attempt (request, response, HTTP status, success/failure)
5. Updates case `external_sync_status` and `external_reference_id`
6. Falls back to local simulation if endpoint is unreachable

---

## SOAP Integration

### Service
A mock SOAP server at `/api/soap-server.php` exposes:

```
verifyCoverage(string clinicName, string insuranceId) → object
```

### Response Object
```
verification_status: "Verified" | "Not Verified"
clinic_approved:     1 | 0
policy_type:         "Full Coverage" | "Partial Coverage" | "Unknown"
message:             string
checked_at:          datetime string
```

### How It Works
1. User clicks "Verify Coverage" on a case detail page
2. `SoapVerificationService` creates a `SoapClient` in non-WSDL mode
3. Calls `verifyCoverage()` on the SOAP server
4. If SOAP server is unreachable, falls back to local simulation
5. Stores the result in `soap_verifications` table
6. Updates the case's `soap_verification_status`

### Running the SOAP Server
To use real SOAP (not simulation), run a second PHP server on port 8001:
```bash
php -S 127.0.0.1:8001 -t .
```

> **Note:** The PHP `soap` extension must be enabled. Check with `php -m | grep soap`.

---

## Security Considerations

| Security Measure | Implementation |
|---|---|
| SQL Injection Prevention | PDO prepared statements with parameter binding |
| XSS Prevention | `htmlspecialchars()` via `h()` helper on all output |
| CSRF Protection | Token-based validation on forms and AJAX (session + header) |
| Password Security | `password_hash(PASSWORD_DEFAULT)` with `password_verify()` |
| Session Hardening | HttpOnly cookies, SameSite=Lax, session regeneration on login |
| Input Sanitization | `sanitizeInput()` trims and strips tags on all user input |
| Error Handling | Graceful user-facing messages; detailed errors logged to DB/file |

---

## Troubleshooting

### Common Issues

**"Database Connection Error" on first visit**
- Run `setup.php` first to create the database
- Verify MySQL is running
- Check credentials in `config/config.php`

**SOAP verification always shows "simulated"**
- Start a second PHP server: `php -S 127.0.0.1:8001 -t .`
- Ensure the `soap` PHP extension is enabled
- Check `SOAP_SERVER_URL` in `config/config.php`

**REST sync fails**
- The app server must be running (cURL calls back to itself)
- With PHP built-in server (single-threaded), uses local simulation fallback
- For real cURL calls, use XAMPP/Apache (multi-threaded)

**"403 Forbidden" on form submission**
- Session may have expired — log in again
- CSRF token mismatch — clear cookies and retry

**Styles not loading**
- Check `BASE_URL` in `config/config.php` matches your server setup
- If using a subdirectory, set `BASE_URL` to `/your-subdirectory/`

---

## QA Checklist

- [x] Login with valid credentials → redirects to dashboard
- [x] Login with invalid credentials → shows error message
- [x] Logout → destroys session, redirects to login
- [x] Unauthenticated access → redirects to login
- [x] Dashboard loads with sample cases and correct badges
- [x] Filters narrow results correctly
- [x] Pagination works across multiple pages
- [x] New case form validates required fields (client-side)
- [x] New case form validates on server-side when JS is bypassed
- [x] CSRF validation prevents cross-site form submission
- [x] AJAX status update changes badge without page reload
- [x] AJAX note addition appears in the notes list instantly
- [x] REST sync logs attempt in integration_logs table
- [x] SOAP verification stores result in soap_verifications table
- [x] Reports page shows correct aggregate counts
- [x] Issues page shows application logs (admin only)
- [x] Non-admin users cannot access Issues page
- [x] All links navigate to correct pages
- [x] Responsive layout works on mobile viewport

---

## Future Enhancements

- File upload support for actual patient images (with HIPAA-compliant storage)
- Role-based assignment workflow with notifications
- Case comments / threaded discussion
- Email notifications for status changes
- PDF report generation
- Audit trail export
- Two-factor authentication
- API key authentication for external systems
- Unit test suite with PHPUnit
- Docker containerization for consistent deployment

---

## Resume Bullets

> Built a PHP-based clinical workflow portal using HTML, CSS, JavaScript, jQuery, AJAX, and MySQL, supporting patient imaging case intake, status management, and role-based access for internal review teams.

> Developed server-rendered intake, dashboard, and reporting pages translating wireframe specifications into functional, validated forms and interactive data views with search, filtering, and pagination.

> Integrated REST and SOAP web services using cURL and PHP SoapClient for external case sync and insurance verification workflows, with full integration logging and retry capabilities.

> Implemented AJAX-driven status updates, support note submission, and real-time integration triggers using jQuery and JSON APIs with CSRF protection and inline error handling.

> Structured a maintainable PHP application architecture with reusable layout partials, PDO-based service classes, secure authentication, application-level logging, and comprehensive technical documentation.

---

*Clinical Image Intake Portal v1.0.0 — Internal Use Only*

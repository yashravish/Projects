const API = {
    async request(method, url, body = null) {
        const opts = { method, headers: { "Content-Type": "application/json" }, credentials: "same-origin" };
        if (body) opts.body = JSON.stringify(body);
        const res = await fetch(url, opts);
        if (res.status === 401) { window.location.href = "/login"; return null; }
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: "Request failed" }));
            throw new Error(err.detail || "Request failed");
        }
        const ct = res.headers.get("content-type") || "";
        return ct.includes("json") ? res.json() : res.text();
    },
    get: (url) => API.request("GET", url),
    post: (url, body) => API.request("POST", url, body),
    patch: (url, body) => API.request("PATCH", url, body),
};

function severityBadge(sev) {
    const s = (sev || "").toLowerCase();
    return `<span class="badge badge-${s}">${sev}</span>`;
}

function statusBadge(st) {
    const s = (st || "").toLowerCase().replace(/ /g, "-");
    return `<span class="badge badge-${s}">${st.replace(/_/g, " ")}</span>`;
}

function riskColor(score) {
    if (score >= 76) return "var(--critical)";
    if (score >= 51) return "var(--danger)";
    if (score >= 26) return "var(--warning)";
    return "var(--success)";
}

function formatDate(d) {
    if (!d) return "—";
    return new Date(d).toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric" });
}

/* ---------- Login ---------- */
async function handleLogin(e) {
    e.preventDefault();
    const username = document.getElementById("username").value;
    const password = document.getElementById("password").value;
    const errEl = document.getElementById("login-error");
    try {
        await API.post("/api/auth/login", { username, password });
        window.location.href = "/dashboard";
    } catch (err) {
        errEl.textContent = err.message;
        errEl.style.display = "block";
    }
}

/* ---------- Dashboard ---------- */
async function loadDashboard() {
    try {
        const d = await API.get("/api/dashboard");
        if (!d) return;
        document.getElementById("stat-vendors").textContent = d.total_vendors;
        document.getElementById("stat-assessments").textContent = d.active_assessments;
        document.getElementById("stat-critical").textContent = d.open_critical_findings;
        document.getElementById("stat-high").textContent = d.open_high_findings;
        document.getElementById("stat-overdue").textContent = d.overdue_remediations;

        renderBarChart("chart-category", d.vendors_by_category, "var(--accent)");
        renderBarChart("chart-severity", d.findings_by_severity, null, { Critical: "var(--critical)", High: "var(--danger)", Moderate: "var(--warning)", Low: "var(--success)" });
        renderBarChart("chart-domain", d.findings_by_domain, "var(--primary-light)");

        const actEl = document.getElementById("recent-activity");
        if (d.recent_activity.length === 0) { actEl.innerHTML = '<p class="empty-state">No activity yet.</p>'; return; }
        actEl.innerHTML = d.recent_activity.map(a =>
            `<div style="padding:8px 0;border-bottom:1px solid var(--border);font-size:12px">
                <strong>${a.action.replace(/_/g, " ")}</strong>
                ${a.details ? "— " + a.details : ""}
                <span style="float:right;color:var(--text-muted)">${formatDate(a.created_at)}</span>
            </div>`
        ).join("");
    } catch (e) { console.error("Dashboard load failed", e); }
}

function renderBarChart(containerId, data, defaultColor, colorMap) {
    const el = document.getElementById(containerId);
    if (!el) return;
    const entries = Object.entries(data || {});
    if (entries.length === 0) { el.innerHTML = '<p class="empty-state" style="height:100%;display:flex;align-items:center;justify-content:center">No data</p>'; return; }
    const max = Math.max(...entries.map(([, v]) => v), 1);
    el.innerHTML = entries.map(([label, val]) => {
        const h = Math.max((val / max) * 110, 4);
        const c = (colorMap && colorMap[label]) || defaultColor || "var(--accent)";
        return `<div class="bar-col"><div class="bar-value">${val}</div><div class="bar" style="height:${h}px;background:${c}"></div><div class="bar-label">${label}</div></div>`;
    }).join("");
}

/* ---------- Vendors ---------- */
async function loadVendors() {
    const data = await API.get("/api/vendors");
    if (!data) return;
    const tbody = document.getElementById("vendors-body");
    if (data.length === 0) { tbody.innerHTML = '<tr><td colspan="6" class="empty-state">No vendors registered.</td></tr>'; return; }
    tbody.innerHTML = data.map(v =>
        `<tr>
            <td><a href="/vendors/${v.id}">${v.name}</a></td>
            <td>${v.category}</td>
            <td>${v.hosting_model || "—"}</td>
            <td>${v.handles_sensitive_data ? "Yes" : "No"}</td>
            <td>${statusBadge(v.status)}</td>
            <td>${formatDate(v.created_at)}</td>
        </tr>`
    ).join("");
}

async function loadVendorDetail(vendorId) {
    const v = await API.get(`/api/vendors/${vendorId}`);
    if (!v) return;
    document.getElementById("vendor-title").textContent = v.name;
    document.getElementById("vendor-info").innerHTML = `
        <div class="grid-2">
            <div><strong>Category:</strong> ${v.category}</div>
            <div><strong>Status:</strong> ${statusBadge(v.status)}</div>
            <div><strong>Business Owner:</strong> ${v.business_owner || "—"}</div>
            <div><strong>Vendor Contact:</strong> ${v.vendor_contact || "—"}</div>
            <div><strong>Hosting Model:</strong> ${v.hosting_model || "—"}</div>
            <div><strong>Deployment Scope:</strong> ${v.deployment_scope || "—"}</div>
            <div><strong>Internet Exposed:</strong> ${v.internet_exposed ? "Yes" : "No"}</div>
            <div><strong>Handles Sensitive Data:</strong> ${v.handles_sensitive_data ? "Yes" : "No"}</div>
            <div><strong>Data Types:</strong> ${(v.data_types || []).join(", ") || "—"}</div>
            <div><strong>Compliance:</strong> ${(v.compliance_attestations || []).join(", ") || "None"}</div>
        </div>
        ${v.description ? `<p style="margin-top:12px;color:var(--text-muted)">${v.description}</p>` : ""}`;

    const integrations = await API.get(`/api/vendors/${vendorId}/integrations`);
    const intEl = document.getElementById("vendor-integrations");
    if (integrations.length === 0) { intEl.innerHTML = "<p>No integrations documented.</p>"; }
    else {
        intEl.innerHTML = `<table><thead><tr><th>System</th><th>Type</th><th>Direction</th><th>Description</th></tr></thead><tbody>` +
            integrations.map(i => `<tr><td>${i.system_name}</td><td>${i.integration_type}</td><td>${i.data_flow_direction}</td><td>${i.description}</td></tr>`).join("") +
            `</tbody></table>`;
    }

    const assessments = await API.get("/api/assessments");
    const vendorAssessments = assessments.filter(a => a.vendor_id === v.id);
    const assEl = document.getElementById("vendor-assessments");
    if (vendorAssessments.length === 0) { assEl.innerHTML = "<p>No assessments yet.</p>"; }
    else {
        assEl.innerHTML = `<table><thead><tr><th>ID</th><th>Type</th><th>Phase</th><th>Risk</th><th>Status</th><th>Actions</th></tr></thead><tbody>` +
            vendorAssessments.map(a => `<tr>
                <td>#${a.id}</td><td>${a.assessment_type}</td><td>${a.phase.replace(/_/g, " ")}</td>
                <td>${a.overall_inherent_risk ? severityBadge(a.overall_inherent_risk) : "—"}</td>
                <td>${statusBadge(a.status)}</td>
                <td>${a.status === "draft" || a.status === "in_progress" ?
                    `<a href="/assessments/${a.id}/questionnaire" class="btn btn-sm btn-secondary">Continue</a>` :
                    `<a href="/assessments/${a.id}/results" class="btn btn-sm btn-secondary">Results</a>`}</td>
            </tr>`).join("") +
            `</tbody></table>`;
    }
}

async function createAssessment(vendorId) {
    const phase = document.getElementById("assessment-phase")?.value || "pre_implementation";
    const type = document.getElementById("assessment-type")?.value || "initial";
    try {
        const a = await API.post("/api/assessments", { vendor_id: vendorId, assessment_type: type, phase });
        window.location.href = `/assessments/${a.id}/questionnaire`;
    } catch (e) { alert("Error: " + e.message); }
}

async function submitVendorForm(e) {
    e.preventDefault();
    const form = e.target;
    const body = {
        name: form.name.value,
        category: form.category.value,
        description: form.description.value,
        website: form.website.value,
        business_owner: form.business_owner.value,
        vendor_contact: form.vendor_contact.value,
        hosting_model: form.hosting_model.value,
        deployment_scope: form.deployment_scope.value,
        internet_exposed: form.internet_exposed.checked,
        handles_sensitive_data: form.handles_sensitive_data.checked,
        data_types: Array.from(form.querySelectorAll('input[name="data_types"]:checked')).map(c => c.value),
        compliance_attestations: Array.from(form.querySelectorAll('input[name="compliance"]:checked')).map(c => c.value),
    };
    try {
        const v = await API.post("/api/vendors", body);
        window.location.href = `/vendors/${v.id}`;
    } catch (e) { alert("Error: " + e.message); }
}

/* ---------- Assessment Questionnaire ---------- */
async function submitAssessment(assessmentId) {
    const form = document.getElementById("questionnaire-form");
    const inputs = form.querySelectorAll("[data-question-key]");
    const answers = [];
    inputs.forEach(el => {
        const key = el.dataset.questionKey;
        const section = el.dataset.section || "";
        const text = el.dataset.questionText || "";
        let val = "";
        if (el.type === "checkbox") val = el.checked ? "true" : "false";
        else if (el.type === "radio") { if (el.checked) val = el.value; else return; }
        else val = el.value;
        if (answers.find(a => a.question_key === key)) return;
        answers.push({ question_key: key, section, question_text: text, answer: val, notes: "" });
    });
    const radios = form.querySelectorAll('input[type="radio"]');
    const radioKeys = new Set();
    radios.forEach(r => {
        const key = r.dataset.questionKey;
        if (radioKeys.has(key)) return;
        radioKeys.add(key);
        const checked = form.querySelector(`input[data-question-key="${key}"]:checked`);
        if (checked && !answers.find(a => a.question_key === key)) {
            answers.push({ question_key: key, section: checked.dataset.section || "", question_text: checked.dataset.questionText || "", answer: checked.value, notes: "" });
        }
    });

    try {
        await API.post(`/api/assessments/${assessmentId}/submit`, { answers });
        const result = await API.post(`/api/assessments/${assessmentId}/evaluate`);
        window.location.href = `/assessments/${assessmentId}/results`;
    } catch (e) { alert("Error: " + e.message); }
}

/* ---------- Assessment Results ---------- */
async function loadAssessmentResults(assessmentId) {
    const a = await API.get(`/api/assessments/${assessmentId}`);
    if (!a) return;
    document.getElementById("result-vendor").textContent = a.vendor_name;
    document.getElementById("result-type").textContent = `${a.assessment_type} — ${a.phase.replace(/_/g, " ")}`;
    document.getElementById("result-status").innerHTML = statusBadge(a.status);

    const score = a.inherent_risk_score || 0;
    document.getElementById("result-inherent-score").textContent = score;
    document.getElementById("result-inherent-rating").innerHTML = a.overall_inherent_risk ? severityBadge(a.overall_inherent_risk) : "—";
    document.getElementById("result-residual-score").textContent = a.residual_risk_score || 0;
    document.getElementById("result-residual-rating").innerHTML = a.overall_residual_risk ? severityBadge(a.overall_residual_risk) : "—";

    const meterFill = document.getElementById("risk-meter-fill");
    if (meterFill) { meterFill.style.width = score + "%"; meterFill.style.background = riskColor(score); }

    document.getElementById("result-summary").textContent = a.executive_summary || "No summary available.";
    const aiEl = document.getElementById("result-ai-summary");
    if (a.ai_summary) { aiEl.textContent = a.ai_summary; aiEl.parentElement.style.display = "block"; }

    const findings = await API.get(`/api/findings?assessment_id=${assessmentId}`);
    document.getElementById("result-findings-count").textContent = findings.length;
    const tbody = document.getElementById("result-findings-body");
    if (findings.length === 0) { tbody.innerHTML = '<tr><td colspan="5" class="empty-state">No findings.</td></tr>'; return; }
    tbody.innerHTML = findings.map(f =>
        `<tr><td>${f.title}</td><td>${severityBadge(f.severity)}</td><td>${f.control_domain_name || "—"}</td><td>${statusBadge(f.remediation_status)}</td><td style="max-width:300px;font-size:12px">${f.recommendation || "—"}</td></tr>`
    ).join("");
}

/* ---------- Findings Dashboard ---------- */
async function loadFindings() {
    const params = new URLSearchParams();
    const sevFilter = document.getElementById("filter-severity")?.value;
    const statusFilter = document.getElementById("filter-status")?.value;
    const domainFilter = document.getElementById("filter-domain")?.value;
    if (sevFilter) params.set("severity", sevFilter);
    if (statusFilter) params.set("status", statusFilter);
    if (domainFilter) params.set("domain", domainFilter);

    const findings = await API.get(`/api/findings?${params}`);
    if (!findings) return;
    const tbody = document.getElementById("findings-body");
    if (findings.length === 0) { tbody.innerHTML = '<tr><td colspan="6" class="empty-state">No findings match filters.</td></tr>'; return; }
    tbody.innerHTML = findings.map(f =>
        `<tr><td>${f.title}</td><td>${severityBadge(f.severity)}</td><td>${f.control_domain_name || "—"}</td><td>${statusBadge(f.remediation_status)}</td><td>${f.owner || "Unassigned"}</td><td>${f.due_date ? formatDate(f.due_date) : "—"}</td></tr>`
    ).join("");
    document.getElementById("findings-count").textContent = findings.length;
}

/* ---------- Remediation Tracker ---------- */
async function loadRemediation() {
    const params = new URLSearchParams();
    const statusFilter = document.getElementById("filter-rem-status")?.value;
    if (statusFilter) params.set("status", statusFilter);

    const items = await API.get(`/api/remediation?${params}`);
    if (!items) return;
    const tbody = document.getElementById("remediation-body");
    if (items.length === 0) { tbody.innerHTML = '<tr><td colspan="7" class="empty-state">No remediation items.</td></tr>'; return; }
    tbody.innerHTML = items.map(r =>
        `<tr>
            <td>${r.finding_title}</td><td>${r.vendor_name || "—"}</td>
            <td>${severityBadge(r.priority)}</td><td>${statusBadge(r.status)}</td>
            <td>${r.assigned_to || "Unassigned"}</td><td>${r.due_date ? formatDate(r.due_date) : "—"}</td>
            <td><button class="btn btn-sm btn-secondary" onclick="openRemEdit(${r.id}, '${r.status}', '${r.assigned_to || ""}', '${r.due_date || ""}')">Edit</button></td>
        </tr>`
    ).join("");
}

function openRemEdit(id, currentStatus, assignedTo, dueDate) {
    const newStatus = prompt("Status (open, in_progress, mitigated, accepted_risk, closed):", currentStatus);
    if (newStatus === null) return;
    const newAssigned = prompt("Assigned to:", assignedTo);
    const newDue = prompt("Due date (YYYY-MM-DD):", dueDate);
    const body = {};
    if (newStatus) body.status = newStatus;
    if (newAssigned !== null) body.assigned_to = newAssigned;
    if (newDue) body.due_date = newDue;
    API.patch(`/api/remediation/${id}`, body).then(() => loadRemediation()).catch(e => alert(e.message));
}

/* ---------- Report Preview ---------- */
async function loadReportPreview(assessmentId) {
    const frame = document.getElementById("report-frame");
    frame.srcdoc = '<p style="padding:40px;text-align:center">Loading report...</p>';
    try {
        const html = await API.request("GET", `/api/reports/${assessmentId}`);
        frame.srcdoc = html;
    } catch (e) {
        frame.srcdoc = `<p style="padding:40px;text-align:center;color:red">${e.message}</p>`;
    }
}

async function generatePDF(assessmentId) {
    try {
        const result = await API.post(`/api/reports/${assessmentId}/generate`);
        alert("Report generated: " + result.file_path);
    } catch (e) { alert("Error: " + e.message); }
}

/* ---------- Governance ---------- */
async function loadGovernance() {
    const templates = await API.get("/api/templates");
    if (!templates) return;
    const tbody = document.getElementById("templates-body");
    tbody.innerHTML = templates.map(t =>
        `<tr><td>${t.name}</td><td>${t.category}</td><td>${t.description}</td><td>${formatDate(t.created_at)}</td></tr>`
    ).join("");

    const domains = await API.get("/api/templates/domains/list");
    const dtbody = document.getElementById("domains-body");
    dtbody.innerHTML = domains.map(d =>
        `<tr><td><strong>${d.code}</strong></td><td>${d.name}</td><td style="font-size:12px">${d.nist_mapping || "—"}</td><td style="font-size:12px">${d.iso_mapping || "—"}</td></tr>`
    ).join("");
}

/* ---------- Logout ---------- */
async function logout() {
    await API.post("/api/auth/logout");
    window.location.href = "/login";
}

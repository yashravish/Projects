/**
 * Centralized Fetch API wrapper for backend communication.
 * All frontend modules use this to make HTTP requests.
 */
const API = (() => {
  const BASE_URL = window.location.origin.includes('8080')
    ? 'http://localhost:8000'
    : window.location.origin;

  async function request(method, path, body = null) {
    const options = {
      method,
      headers: { 'Content-Type': 'application/json' },
    };
    if (body) {
      options.body = JSON.stringify(body);
    }
    const response = await fetch(`${BASE_URL}${path}`, options);
    const data = await response.json();
    if (!response.ok) {
      const message = data.detail || JSON.stringify(data);
      throw new Error(message);
    }
    return data;
  }

  return {
    get:  (path)       => request('GET', path),
    post: (path, body) => request('POST', path, body),

    getHealth:          ()     => request('GET', '/api/health'),
    getDashboardSummary:()     => request('GET', '/api/dashboard/summary'),
    getDeviceStatus:    ()     => request('GET', '/api/device/status'),
    getCaptures:        ()     => request('GET', '/api/captures'),
    getCapture:         (id)   => request('GET', `/api/captures/${id}`),
    createCapture:      (data) => request('POST', '/api/captures', data),
    retryCapture:       (id)   => request('POST', `/api/captures/${id}/retry`),
    getDefects:         ()     => request('GET', '/api/defects'),
    createDefect:       (data) => request('POST', '/api/defects', data),
  };
})();

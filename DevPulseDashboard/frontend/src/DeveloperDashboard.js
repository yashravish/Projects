// frontend/src/DeveloperDashboard.js
import React, { useEffect, useState } from 'react';
import axios from 'axios';

const DeveloperDashboard = () => {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    axios.get('/api/metrics')
      .then(response => {
        setMetrics(response.data);
        setLoading(false);
      })
      .catch(err => {
        setError('Error fetching metrics data');
        setLoading(false);
      });
  }, []);

  if (loading) return <div>Loading metrics...</div>;
  if (error) return <div>{error}</div>;

  return (
    <div style={{ padding: '20px' }}>
      <h2>Developer Productivity Analytics Dashboard</h2>
      <p>Last updated: {metrics.timestamp}</p>
      <table border="1" cellPadding="10" style={{ borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            <th>Metric</th>
            <th>Value</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>PR Turnaround Time (hours)</td>
            <td>{metrics.pr_turnaround_time_hours}</td>
          </tr>
          <tr>
            <td>Test Pass Rate (%)</td>
            <td>{metrics.test_pass_rate_percent}</td>
          </tr>
          <tr>
            <td>Average Jira Issue Resolution (days)</td>
            <td>{metrics.jira_issue_resolution_days}</td>
          </tr>
          <tr>
            <td>Active Deployments</td>
            <td>{metrics.active_deployments}</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
};

export default DeveloperDashboard;

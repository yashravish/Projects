// react-dashboard/src/CoverageDashboard.js
import React, { useEffect, useState } from 'react';
import axios from 'axios';

const CoverageDashboard = () => {
  const [coverageData, setCoverageData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Replace with your actual API endpoint URL that returns coverage data
    axios.get('/api/coverage')
      .then(response => {
        setCoverageData(response.data);
        setLoading(false);
      })
      .catch(err => {
        console.error('Error fetching coverage data:', err);
        setError('Failed to load coverage data.');
        setLoading(false);
      });
  }, []);

  if (loading) return <div>Loading test coverage data...</div>;
  if (error) return <div>{error}</div>;

  return (
    <div>
      <h2>Test Coverage Dashboard</h2>
      <table border="1" cellPadding="8" style={{ borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            <th>Microservice</th>
            <th>Coverage (%)</th>
          </tr>
        </thead>
        <tbody>
          {coverageData.map((service, index) => (
            <tr key={index}>
              <td>{service.name}</td>
              <td>{service.coverage}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default CoverageDashboard;

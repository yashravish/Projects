import React from 'react';
import CodeQualityDashboard from './components/CodeQualityDashboard';
import './App.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>CodeSentinel Dashboard</h1>
        <p className="subtitle">Code Quality Monitoring Platform</p>
      </header>
      
      <main className="dashboard-container">
        <CodeQualityDashboard />
      </main>

      <footer className="App-footer">
        <p> {new Date().getFullYear()} CodeSentinel - Automated Code Review System</p>
      </footer>
    </div>
  );
}

export default App;
// src/components/LandingPage.js
import React from 'react';

const LandingPage = () => {
  return (
    <header className="landing-header">
      <div className="container">
        <h1>Welcome to BizMarketing</h1>
        <p>Your one‑stop solution for comprehensive marketing across platforms.</p>
        <a href="/services" className="btn">
          Explore Our Services
        </a>
      </div>
    </header>
  );
};

export default LandingPage;

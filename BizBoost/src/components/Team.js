// src/components/Team.js
import React from 'react';

const Team = () => {
  return (
    <section className="team-section">
      <div className="container">
        <h2>Meet Our Team</h2>
        <div className="team-list">
          <div className="team-member">
            <img src="https://via.placeholder.com/150" alt="Alice Johnson" />
            <h3>Alice Johnson</h3>
            <p>Marketing Director</p>
          </div>
          <div className="team-member">
            <img src="https://via.placeholder.com/150" alt="Bob Smith" />
            <h3>Bob Smith</h3>
            <p>Digital Strategist</p>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Team;

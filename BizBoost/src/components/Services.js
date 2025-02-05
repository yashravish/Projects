// src/components/Services.js
import React from 'react';

const Services = () => {
  return (
    <section className="services-section">
      <div className="container">
        <h2>Our Services</h2>
        <div className="services-list">
          <div className="service-item">
            <h3>Digital Strategy</h3>
            <p>We create robust digital strategies that drive results.</p>
          </div>
          <div className="service-item">
            <h3>SEO & SEM</h3>
            <p>Improve your online visibility and drive organic traffic.</p>
          </div>
          <div className="service-item">
            <h3>Social Media Marketing</h3>
            <p>Engage with your audience and grow your brand on social media.</p>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Services;

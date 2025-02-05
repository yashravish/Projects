// src/components/Testimonials.js
import React from 'react';

const Testimonials = () => {
  return (
    <section className="testimonials-section">
      <div className="container">
        <h2>Client Testimonials</h2>
        <div className="testimonial-list">
          <div className="testimonial-item">
            <p>"BizMarketing transformed our business. Highly recommended!"</p>
            <h4>– John Doe, CEO of Acme Inc.</h4>
          </div>
          <div className="testimonial-item">
            <p>"Our sales have skyrocketed thanks to their innovative strategies."</p>
            <h4>– Jane Smith, Founder of Startup XYZ</h4>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Testimonials;

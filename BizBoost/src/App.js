// src/App.js
import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import LandingPage from './components/LandingPage';
import Services from './components/Services';
import Testimonials from './components/Testimonials';
import CaseStudies from './components/CaseStudies';
import Team from './components/Team';
import ContactForm from './components/ContactForm';
import ChatWidget from './components/ChatWidget';
import Footer from './components/Footer';

function App() {
  return (
    <div>
      <Navbar />
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/services" element={<Services />} />
        <Route path="/testimonials" element={<Testimonials />} />
        <Route path="/case-studies" element={<CaseStudies />} />
        <Route path="/team" element={<Team />} />
        <Route path="/contact" element={<ContactForm />} />
      </Routes>
      <ChatWidget />
      <Footer />
    </div>
  );
}

export default App;

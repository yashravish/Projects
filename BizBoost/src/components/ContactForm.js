// src/components/ContactForm.js
import React from 'react';
import { useForm } from 'react-hook-form';

const ContactForm = () => {
  const {
    register,
    handleSubmit,
    formState: { errors },
    reset
  } = useForm();

  const onSubmit = (data) => {
    console.log(data);
    // Here you could integrate with your CRM or email marketing API.
    alert('Your message has been sent!');
    reset();
  };

  return (
    <section className="contact-section">
      <div className="container">
        <h2>Contact Us</h2>
        <form onSubmit={handleSubmit(onSubmit)} className="contact-form">
          <div className="form-group">
            <label htmlFor="name">Name</label>
            <input
              type="text"
              id="name"
              {...register('name', { required: true })}
            />
            {errors.name && (
              <span className="error">This field is required</span>
            )}
          </div>
          <div className="form-group">
            <label htmlFor="email">Email</label>
            <input
              type="email"
              id="email"
              {...register('email', {
                required: true,
                pattern: /^\S+@\S+$/i
              })}
            />
            {errors.email && (
              <span className="error">
                Please enter a valid email
              </span>
            )}
          </div>
          <div className="form-group">
            <label htmlFor="message">Message</label>
            <textarea
              id="message"
              {...register('message', { required: true })}
            ></textarea>
            {errors.message && (
              <span className="error">This field is required</span>
            )}
          </div>
          <button type="submit" className="btn">
            Send Message
          </button>
        </form>
      </div>
    </section>
  );
};

export default ContactForm;

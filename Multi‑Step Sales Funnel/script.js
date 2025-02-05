document.addEventListener('DOMContentLoaded', function() {
    // --- Lead Magnet Email Capture with Simulated Double Opt-In ---
    const optinForm = document.getElementById('optin-form');
    if (optinForm) {
      optinForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const emailInput = document.getElementById('email');
        const messageDiv = document.getElementById('form-message');
        const email = emailInput.value.trim();
  
        if (email) {
          // Simulate sending a double opt-in email
          messageDiv.innerHTML = `<p>Thank you! Please check your email (<strong>${email}</strong>) to confirm your subscription and receive your checklist.</p>`;
          // In a real implementation, an AJAX call to your email automation backend would be made here.
          optinForm.reset();
        }
      });
    }
  
    // --- Pricing Page: Handle Plan Selection and Upsell Modal ---
    window.selectPlan = function(plan) {
      // Store selected plan in localStorage or pass it via URL parameters (for demo purposes, we simply log it)
      console.log(`Plan selected: ${plan}`);
      // Optionally show an upsell modal for an order bump
      showUpsellModal(plan);
    };
  
    function showUpsellModal(plan) {
      const upsellModal = document.getElementById('upsell-modal');
      if (upsellModal) {
        upsellModal.style.display = 'block';
        // You could further customize the modal based on the selected plan if needed
      }
    }
  
    // Close modal logic
    const closeModalBtn = document.getElementById('close-modal');
    if (closeModalBtn) {
      closeModalBtn.addEventListener('click', function() {
        document.getElementById('upsell-modal').style.display = 'none';
        // Redirect to thank you page after upsell flow completes or is skipped
        window.location.href = 'thankyou.html';
      });
    }
    
    // Upsell Accept Button Logic
    const acceptUpsellBtn = document.getElementById('accept-upsell');
    if (acceptUpsellBtn) {
      acceptUpsellBtn.addEventListener('click', function() {
        // Process the upsell offer (e.g., add order bump to purchase)
        console.log('Upsell accepted');
        // Close modal and redirect to thank you page
        document.getElementById('upsell-modal').style.display = 'none';
        window.location.href = 'thankyou.html';
      });
    }
  
    // Optionally, add logic for abandoned cart recovery using cookies or localStorage.
    // For example, if a user leaves the page with items in their "cart", you could trigger a reminder.
  });
  
import wixData from 'wix-data';

export function commissionForm_submit(event) {
  event.preventDefault(); // Prevent default form submission behavior

  // Create an object to insert into the CommissionRequests collection
  let requestData = {
    artistId: $w('#artistIdInput').value,
    name: $w('#nameInput').value,
    email: $w('#emailInput').value,
    details: $w('#detailsInput').value,
    requestedDate: $w('#datePicker').value
  };

  wixData.insert("CommissionRequests", requestData)
    .then((result) => {
      $w("#formStatus").text = "Thank you! Your commission request has been submitted.";
      // Optionally, clear the form inputs:
      $w('#artistIdInput, #nameInput, #emailInput, #detailsInput, #datePicker').forEach(input => input.value = "");
    })
    .catch((error) => {
      console.error("Commission request error:", error);
      $w("#formStatus").text = "There was an error submitting your request. Please try again later.";
    });
}

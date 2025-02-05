// Import the wix-data API to query your Artworks collection
import wixData from 'wix-data';

$w.onReady(function () {
  // Query the "Artworks" collection
  wixData.query("Artworks")
    .limit(50) // Adjust limit as needed
    .find()
    .then((results) => {
      if (results.items.length > 0) {
        // Bind the results to the repeater
        $w("#repeaterGallery").data = results.items;
      }
    })
    .catch((err) => {
      console.error("Error retrieving artworks:", err);
    });

  // Lazy load images using IntersectionObserver (for modern browsers)
  const repeater = $w("#repeaterGallery");
  repeater.onItemReady(($item, itemData, index) => {
    // Set a placeholder image first
    $item("#artworkImage").src = "https://static.wixstatic.com/media/placeholder.jpg";
    
    // Use a timeout or IntersectionObserver (if supported) to load the real image
    setTimeout(() => {
      $item("#artworkImage").src = itemData.image;
      // Optionally add a fade-in animation once the image loads
      $item("#artworkImage").show("fade", { duration: 800 });
    }, 200 * index); // stagger loading by index
     
    // Set artwork title text
    $item("#artworkTitle").text = itemData.title;
  });
});

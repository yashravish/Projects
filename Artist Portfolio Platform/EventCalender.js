import wixData from 'wix-data';

$w.onReady(function () {
  // Query the "Events" collection for upcoming events
  wixData.query("Events")
    .ge("startDate", new Date()) // events starting from today
    .limit(20)
    .ascending("startDate")
    .find()
    .then((results) => {
      if (results.items.length > 0) {
        $w("#repeaterEvents").data = results.items;
      } else {
        $w("#noEventsMessage").text = "No upcoming sessions. Please check back later.";
      }
    })
    .catch((err) => {
      console.error("Error loading events:", err);
    });

  $w("#repeaterEvents").onItemReady(($item, itemData) => {
    $item("#eventTitle").text = itemData.title;
    $item("#eventDate").text = new Date(itemData.startDate).toLocaleDateString();
    $item("#eventDescription").text = itemData.description;
  });
});

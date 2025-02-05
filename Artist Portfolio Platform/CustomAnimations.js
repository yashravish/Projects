// In the onItemReady for the gallery, you can add a hover effect:
$w("#artworkImage").onMouseIn(() => {
    $w("#artworkImage").expand(); // Or use wix-window animations if needed
  });
  
  $w("#artworkImage").onMouseOut(() => {
    $w("#artworkImage").collapse();
  });
  
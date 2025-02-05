import wixData from 'wix-data';

$w.onReady(function () {
  wixData.query("BlogPosts")
    .limit(10)
    .descending("publishDate")
    .find()
    .then((results) => {
      if (results.items.length > 0) {
        $w("#repeaterBlog").data = results.items;
      }
    })
    .catch((err) => {
      console.error("Error loading blog posts:", err);
    });

  $w("#repeaterBlog").onItemReady(($item, itemData) => {
    $item("#postTitle").text = itemData.title;
    $item("#postExcerpt").text = itemData.content.substring(0, 150) + "...";
    $item("#postCover").src = itemData.coverImage;
  });
});

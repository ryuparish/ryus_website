---
layout: page
title: "butterfly"
description: No more wondering if you can make your "usual" outfit
img: assets/img/RyuCVDevice.jpeg
importance: 1
category: Recreational CS
---

I noticed that when I do laundry I would, at some point, sort of "display" the article of clothing before folding
the laundry like this: 

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/folding_laundry.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    *Not me*
</div>

That's when I realized, I could probably take a picture right when I did this by either detecting this pose and then
automatically taking a picture, or more chaotically (this is what I did), by setting a small device to take a photo
every 15 seconds or so until my laundry was all folded.

First, I would take a picture of each article of clothing in my closet (this was the most laborous part, but it gets very
convenient afterwards).

Then, I could use a CNN/Siamese network to match (or at least match the general class of the article of clothing)
the currently-being-folded piece of clothing to an existing article of clothing and mark this piece of clothing as clean.

Finally, on a daily basis, when I would choose my outfit I could just take a single photo, mark all the clothing I am wearing
as "dirty" and have an actively tracked wardrobe application. With this, I could always know what outfit I could wear, approximately how long until no outfits that I usually wore availble (time to do the laundry).

Here is the device that I used (a cheap chasis, Nvidia Jetson Nano 2GB, and high quality camera):

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/RyuCVDevice.jpeg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    I actually had a different camera at first, but the quality created a training-data / testing-data mismatch.
</div>

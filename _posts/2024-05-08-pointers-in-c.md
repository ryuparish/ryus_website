---
layout: post
title: How I understand pointers in C.
date: 2022-01-04 11:59:00-0400
description: I try to explain my understanding of pointers in C.
categories: How-To
---

I have a hard time learning certain things in computer science. Pointers in C was one of those things that took many re-learnings to understand since there are so many ways to misinterpret how they work.

I also have a mental representation in my mind that helped me never forget how deferencing and addresses in C work:

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/pointer_explanation.jpg" title="An image of a number attached to a spring with an address on it" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    My drawing of my mental image.
</div>

This is representation of the variable: `int x = 8;`. The spring is the address and 8 is the value.

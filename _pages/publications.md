---
layout: page
permalink: /publications/
title: publications
description: My current publications
years: [2022]
nav: true
---
<!-- _pages/publications.md -->
<div class="publications">

<!-- commented out since I have no papers! -->
{%- for y in page.years %}
  <h2 class="year">{{y}}</h2>
  {% bibliography -f papers -q @*[year={{y}}]* %}
{% endfor %}

</div>

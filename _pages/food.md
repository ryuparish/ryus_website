---
layout: page
title: food
permalink: /food/
description: I love to cook. All are from scratch.
nav: true
display_categories: [Pastries, Japanese, Batter Breads]
horizontal: false
---

<center>
<div class="row">
  <div class="col-sm mt-3 mt-md-0">
      {% include figure.html path="assets/img/Ryu_Nutella.jpeg" title="example image" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
</center>

<!-- pages/projects.md -->
<div class="projects">
<!-- Display categorized projects -->
{%- for category in page.display_categories %}
<h2 class="category">{{ category }}</h2>
{%- assign categorized_projects = site.projects | where: "category", category -%}
{%- assign sorted_projects = categorized_projects | sort: "importance" %}
<!-- Generate cards for each project -->
{% if page.horizontal -%}
<div class="container">
  <div class="row row-cols-2">
  {%- for project in sorted_projects -%}
    {% include projects_horizontal.html %}
  {%- endfor %}
  </div>
</div>
{%- else -%}
<div class="grid">
  {%- for project in sorted_projects -%}
    {% include projects.html %}
  {%- endfor %}
</div>
{%- endif -%}
{% endfor %}
</div>

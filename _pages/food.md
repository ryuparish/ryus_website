---
layout: page
title: food
permalink: /food/
description: I love to cook.
nav: true
display_categories: [Pies, Japanese]
horizontal: false
---

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
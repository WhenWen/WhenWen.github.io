---
layout: archive
title: "Publications"
permalink: /publications/
author_profile: true
---

For proper citation, refer 

{% include base_path %}

{% for post in site.publications reversed %}
  {% include archive-single.html %}
{% endfor %}

---
layout: post
title: "From Weight Decay to Hyperball Optimization"
description: "How weight decay shapes the effective learning rate and how Hyperball optimizer replaces it."
date: 2025-11-30
redirect: /wd_blog/public/index.html
tags:
  - weight decay
  - hyperball
  - optimization
  - deep learning
categories:
  - research
thumbnail: /wd_blog/assets/images/fig0.png
giscus_comments: false
toc: false
---

<script>
  if (typeof window !== 'undefined') {
    window.location.replace('{{ "/wd_blog/public/index.html" | relative_url }}');
  }
</script>

Weight decay is often described as capacity control, but in modern scale-invariant architectures it instead sets the *effective* step size. Xingyu, Kaifeng, Tengyu, Percy, and I put together a full-length interactive article that walks through the math, demos, and Hyperball — an optimizer that removes weight decay entirely by constraining norms directly.

All of the interactive plots, sliders, and citations live in the standalone build below. You can read it inline or pop it out into a new tab if you want a full-screen view.

<div class="wd-blog-frame">
  <div class="wd-blog-frame__toolbar">
    <a class="btn" href="{{ '/wd_blog/public/index.html' | relative_url }}" target="_blank" rel="noopener">
      Open full page
    </a>
  </div>
  <iframe
    id="wd-blog-iframe"
    src="{{ '/wd_blog/public/index.html' | relative_url }}"
    title="From Weight Decay to Hyperball Optimization"
    loading="lazy"
    referrerpolicy="strict-origin-when-cross-origin"
    style="width: 100%; border: 1px solid var(--global-muted-color, #e5e7eb); border-radius: 8px; min-height: 1200px;"
  ></iframe>
</div>

<script>
  (function () {
    const iframe = document.getElementById('wd-blog-iframe');
    if (!iframe) return;
    const resize = () => {
      try {
        const doc = iframe.contentDocument || iframe.contentWindow.document;
        if (!doc || !doc.body) return;
        const height = doc.body.scrollHeight;
        if (height > 0) {
          iframe.style.height = `${height + 100}px`;
        }
      } catch (err) {
        console.warn('wd_blog iframe resize failed', err);
      }
    };
    iframe.addEventListener('load', resize);
    window.addEventListener('resize', () => {
      window.requestAnimationFrame(resize);
    });
  })();
</script>

<style>
  .wd-blog-frame {
    margin-top: 1.5rem;
    margin-bottom: 2.5rem;
  }
  .wd-blog-frame__toolbar {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 0.5rem;
  }
  .wd-blog-frame__toolbar .btn {
    border-radius: 999px;
    font-weight: 600;
    padding: 0.4rem 1.2rem;
  }
</style>


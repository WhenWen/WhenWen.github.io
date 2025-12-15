---
layout: post
title: "From Weight Decay to Hyperball Optimization (Series)"
description: "Two-part interactive series: Hyperball optimizer + a theory deep dive on weight decay and effective step size."
date: 2025-11-30
published: false
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
read_time: 40
---

This is a two-part interactive series:

- **Part 1 (Hyperball optimizer + intuition)**: [`/wd_blog/public/hyperball-part-1.html`]( {{ '/wd_blog/public/hyperball-part-1.html' | relative_url }} )
- **Part 2 (theory deep dive on weight decay & effective step size)**: [`/wd_blog/public/weight-decay-part-2.html`]( {{ '/wd_blog/public/weight-decay-part-2.html' | relative_url }} )

Weight decay is often described as capacity control, but in modern scale-invariant architectures it instead sets the *effective* step size. Xingyu, Kaifeng, Tengyu, Percy, and I put together two standalone interactive articles with the math, demos, and Hyperball—an optimizer that removes weight decay entirely by constraining norms directly.

If you want the full interactive experience, open the pages directly (links above).


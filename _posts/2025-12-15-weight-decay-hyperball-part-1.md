---
layout: post
title: "From Weight Decay to Hyperball Optimization (Part 1): Hyperball optimizer + intuition"
description: "Part 1 of a two-part series: Hyperball optimization and empirical intuition."
date: 2025-12-14
tags:
  - weight decay
  - hyperball
  - optimization
  - deep learning
categories:
  - research
thumbnail: /wd_blog/assets/images/fig0.png
giscus_comments: false
toc:
  sidebar: left
read_time: 25
---

> **Authors:** [Kaiyue Wen](https://whenwen.github.io/), [Xingyu Dang](https://dangxingyu.github.io/), [Kaifeng Lyu](https://kaifeng.ac/), [Tengyu Ma](https://ai.stanford.edu/~tengyuma/), [Percy Liang](https://cs.stanford.edu/~pliang/)

---

**Series navigation**

- **Part 1 (this page):** Hyperball optimizer + intuition
- **Part 2 (theory deep dive):** [Deep Dive into Theory of Weight Decay and how it sets the effective step size](/blog/2025/weight-decay-hyperball-part-2/)

---

## Hyperball Optimization: Normalizing Both Weight Norm and Update Norm

Recent deep learning theory shows that weight decay is just a proxy for controlling $\|W\|_F$. The weight norm further determines the effective step size: how large the update is *relative* to the weight (see Section 2 and deep dive in Part 2).

In light of this theory, we propose **Hyperball Optimization**:

$$
W_{t+1} = \mathrm{Norm}(W_t - \eta_{\mathrm{eff}} \cdot \mathrm{Norm}(u_t)\,\|W_0\|_F)\,\|W_0\|_F,
$$

where $\|W_0\|_F$ is the initial weight norm and $\mathrm{Norm}(x) = x / \|x\|_F$ projects $x$ to the unit sphere. Here $u_t$ is the standard Adam/Muon update. We apply this optimizer to each Transformer projection matrix and use Adam for the remaining scalar parameters and embeddings. We call the corresponding optimizer **Adam-Hyperball (AdamH)** or **Muon-Hyperball (MuonH)** depending on the base optimizer.

This idea is closely related to Weight Normalization [22], though we no longer maintain a separate norm parameter. Variants that omit update-norm normalization have been explored in diffusion models [23] and, concurrently with our work, in language model training [24]. We do not expect update-norm normalization to deliver a large empirical speedup, but it is crucial for native hyperparameter transfer because it preserves the ratio between the weight and update norms.

> **📊 Interactive Demo:** *W&B Metrics Plot - Hyperball Runs*
> 
> `[Interactive visualization available in the HTML version]`
> 
> See the [interactive article](/wd_blog/public/hyperball-part-1.html) for the full demo.

This method has the following benefits:
1. **Native hyperparameter transfer:** Hyperparameters transfer predictably across model width and depth.
2. **Empirical speedup:** Faster loss reduction compared to decoupled weight decay across model scale and training duration.

### Hyperparameter Transfer across Width and Depth

According to recent deep learning theory (see details in Phenomenon 5 in Part 2), hyperparameter transfer should be a byproduct of keeping the ratio between the **weight norm** and the **update norm** essentially constant across model scales. With normalization, loss does not "care" the raw numbers inside the weights; it sees how big each update is **relative** to the current weights, which in turn determines how far the model moves in function space per step. If widening or deepening the network changes this ratio, then the same learning rate suddenly becomes "too hot" or "too cold," and the optimal learning-rate window drifts with scale.

Hyperball optimization fixes this by explicitly controlling both the weight norm and the update norm. Once we normalize $W$ and $u_t$ and then rescale by a chosen reference norm $\|W_0\|_F$, the ratio

$$
\frac{\|u_t\|_F}{\|W_t\|_F}
$$

is no longer a subtle consequence of initialization (e.g. MuP initialization) and architecture—it becomes a **designed quantity**. In other words, the "effective step size" on the unit sphere is engineered to be stable across widths and depths, so the same nominal $(\eta_{\text{eff}}, T)$ schedule produces comparable trajectories for a 4-layer toy model and a 512-layer model.

This perspective lines up with the **spectral condition** view of feature learning: what matters is the effective step size along each eigen-direction of the data/feature covariance, not the raw learning rate alone. Yang, Simon, and Bernstein show that feature learning occurs when the product of step size and appropriate spectral quantities lies in a "sweet spot" that is neither too small (no learning) nor too large (instability) [25]. By stabilizing the ratio between weight and update norms, Hyperball makes it much easier to keep the model inside this sweet spot across architectures—so the same hyperparameters induce similar spectral dynamics even as we scale width and depth.

We validate this by running the following width and depth scaling experiments:

**1. Depth Scaling:** We fix the number of hidden dimensions to be 128 and vary the number of layers from 4 to 512. We observe that the maximal drift of optimal learning rate window is 1.4x, where as the drift is 3x for Adam and 4x for Muon.

> **📊 Interactive Demo:** *Depth Scaling Plot*
> 
> `[Interactive visualization available in the HTML version]`
> 
> See the [interactive article](/wd_blog/public/hyperball-part-1.html) for the full demo.

**2. Width Scaling:** We fix the number of layers to be 4 and vary the number of hidden dimensions from 128 to 2048. We observe that the maximal drift of optimal learning rate window is 1.4x for AdamH and MuonH, whereas the drift is 2x for Adam and 4x for Muon.

> **📊 Interactive Demo:** *Width Scaling Plot*
> 
> `[Interactive visualization available in the HTML version]`
> 
> See the [interactive article](/wd_blog/public/hyperball-part-1.html) for the full demo.

### Empirical Speedup 

We further validate Hyperball in the same setup as [Marin's speedrun](https://marin.community/speedrun/) and observe that Hyperball leads to empirical speedup over AdamW and MuonW using a Gemma-like architecture.

> **📊 Interactive Demo:** *C4/en Loss Plot*
> 
> `[Interactive visualization available in the HTML version]`
> 
> See the [interactive article](/wd_blog/public/hyperball-part-1.html) for the full demo.

Further, we heavily overtrained a small 130M model with MuonH and Muon and observe a consistent speedup across different data sizes.

> **📊 Interactive Demo:** *Data Scaling Plot*
> 
> `[Interactive visualization available in the HTML version]`
> 
> See the [interactive article](/wd_blog/public/hyperball-part-1.html) for the full demo.

---

## Why Hyperball? A Secret Life of Weight Decay

Before Hyperball, the common wisdom is to use weight decay to control the weight norm. Weight decay is often tossed in under the assumption that it acts as a proxy for L2 regularization to prevent overfitting. But here is the twist: when you pair it with normalization layers (like LayerNorm), weight norm actually stops affecting model capacity entirely. So, why does weight decay still work?

It turns out that weight decay has been living a double life. Recent deep learning theory (see a deep dive in Part 2) shows that weight decay actually determines the **equilibrium weight norm**, and that norm determines how large a step the optimizer takes in *direction space*—i.e., the **effective step size** (with the rule of thumb $\eta_{\mathrm{eff}} \propto \sqrt{\eta\lambda}$). In contrast, without weight decay, the weight norm will grow as $\Theta(\sqrt{t})$, so the effective step size decays naturally: $\eta_{\mathrm{eff}} \propto \sqrt{1/t}$. Therefore, weight decay prevents the direction-space step size from decaying too quickly, inducing an effective schedule that often decays **more slowly** than "no weight decay," so you keep making useful progress later in training [33][34][35].

> **📊 Interactive Demo:** *W&B QKV Norms Plot - Analysis*
> 
> `[Interactive visualization available in the HTML version]`
> 
> See the [interactive article](/wd_blog/public/hyperball-part-1.html) for the full demo.

---

## Commonly Asked Questions

<details>
<summary><strong>Q1: Why does Hyperball speed up training compared to decoupled weight decay?</strong></summary>

Training with decoupled weight decay with a slowly changing learning rate will converge to a fixed weight norm and induce an 'implicit' learning rate schedule. Hyperball makes the effective step-size schedule explicit and this schedule turns out to be better.

As we will show in details in Part 2, training with decoupled weight decay with a slowly changing learning rate will converge to a fixed weight norm that is a function of the optimizer choice and hyperparameters. This induces an 'implicit' learning rate schedule. However, this implicit schedule is hard to reason about as in practice, one need to account for the changing learning rate and this introduces a lagging effect on weight norm.

Hyperball makes the effective step-size schedule explicit instead of implicit and the explicit schedule turns out to be better. Our finding is also consistent with [32], which argues that hyperparameter transfer can enable consistent gains of matrix-preconditioned optimizers across scale.

</details>

<details>
<summary><strong>Q2: Will weight normalization hurt representation power?</strong></summary>

No, in most cases. RMSNorm's trainable affine term keeps the model expressive even with fixed weight norms.

The commonly used RMSNorm affine transformation provides an explicit way to control the norm. Concretely, pre-norm neural networks consist of linear projections that look like

$$
f(h; W, \gamma) = W (\gamma \odot \mathrm{RMSNorm}(h))
$$

where $W$ is the weight matrix, $\gamma$ is the scaling vector, and $\mathrm{RMSNorm}(h)$ is the RMSNorm normalization. Because $\gamma$ is trainable, fixing the norm of $W$ doesn't hurt representation power as we have

$$
f(h; cW, \gamma/c) = f(h; W, \gamma).
$$

</details>

<details>
<summary><strong>Q3: Why don't we control the spectral norm instead of the Frobenius norm?</strong></summary>

In practice, the two norms track each other because transformer weight matrices stay close to full rank, so constraining $\|W\|_F$ already constrains $\|W\|_{\mathrm{op}}$.

If you focus on steepest-descent methods such as Muon and related spectral condition theory [25], it feels more natural to constrain the spectral norm. That intuition is correct for update matrices. Here, however, we are constraining the *weight* matrices themselves, and doing so through the Frobenius norm is almost equivalent when those matrices remain well-conditioned.

Assume a singular value decomposition $W = U \Sigma V^\top$ with $\Sigma = \mathrm{diag}(\sigma_1, \ldots, \sigma_d)$. Then

$$
\|W\|_F^2 = \sum_{i=1}^d \sigma_i^2, \qquad \|W\|_{\mathrm{op}}^2 = \sigma_1^2.
$$

The ratio

$$
r = \frac{\|W\|_F^2}{d \|W\|_{\mathrm{op}}^2} = \frac{\sum_{i=1}^d \sigma_i^2}{d \sigma_1^2} \in \left[\frac{1}{d}, 1\right]
$$

measures how concentrated the singular values are. Low-rank matrices push $r$ toward $1/d$, while full-rank matrices keep it near $1$.

In the deep-learning regime of interest, weight matrices empirically hover near full rank, so $r$ stays far from $0$ [14]. That makes Frobenius-norm control an excellent proxy for spectral-norm control without needing additional machinery [26][27][28][29][30].

![Empirical SVD of weight matrix](/wd_blog/assets/images/svd_dist_attn_final_v2.png)
*Empirical SVD of weight matrix. Figure from [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982) [14].*

Further, theoretically, [31] shows that Muon with decoupled weight decay implicitly optimize weights under spectral norm constraints. As we show in Section 2, our method corresponds to Muon with decoupled weight decay with a special effective step size schedule, and therefore we expect it to have similar behavior.

</details>

---

## Acknowledgments

The authors would like to thank [Songlin Yang](https://sustcsonglin.github.io/), [Zihan Qiu](https://www.linkedin.com/in/zihan-qiu-33a172249/), and [Liliang Ren](https://renll.github.io/) for motivating this blog post into existence. To some extent, this work is a proof to show that it is possible to remove weight decay altogether by designing the optimizer to explicitly control weight norms. The authors would also like to thank [William Held](https://williamheld.com/), [David Hall](http://dlwh.org/), [Suhas Kotha](https://kothasuhas.github.io/), [Tatsunori Hashimoto](https://thashim.github.io/), [Jason Lee](https://jasondlee88.github.io/), [Zhiyuan Li](https://zhiyuanli.ttic.edu/), [Lijie Chen](https://chen-lijie.github.io/), [Huaqing Zhang](https://scholar.google.com/citations?user=_E9tcTkAAAAJ), [Jiacheng You](https://github.com/YouJiacheng), [Jeremy Bernstein](https://jeremybernste.in/) and [Samuel Schoenholz](https://www.linkedin.com/in/samuel-schoenholz-379830a0/) for helpful discussions.

---

## Citations

If this work is helpful to you, please consider citing:

```bibtex
@online{wen2025hyperball_part1,
    title        = {Fantastic Pretraining Optimizers and Where to Find Them II: From Weight Decay to Hyperball Optimization},
    author       = {Wen, Kaiyue and Dang, Xingyu and Lyu, Kaifeng and Ma, Tengyu and Liang, Percy},
    year         = {2025},
    month        = {12},
    day          = {15},
    url          = {https://whenwen.github.io/wd_blog/public/hyperball-part-1.html},
    urldate      = {2025-12-15},
}
```

---

## References

1. **Jingyuan Liu, Jianlin Su, Xingcheng Yao, et al.** "Muon is Scalable for LLM Training." (2025). [https://arxiv.org/abs/2502.16982](https://arxiv.org/abs/2502.16982)

2. **Tim Salimans, Diederik P. Kingma**. "Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks." (2016). [https://arxiv.org/abs/1602.07868](https://arxiv.org/abs/1602.07868)

3. **Tero Karras, Miika Aittala, et al.** "Analyzing and Improving the Training Dynamics of Diffusion Models." (2023). [https://arxiv.org/abs/2312.02696](https://arxiv.org/abs/2312.02696)

4. **Yonggan Fu, Xin Dong, et al.** "Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models." (2025). [https://arxiv.org/abs/2511.18890](https://arxiv.org/abs/2511.18890)

5. **Greg Yang, James B. Simon, Jeremy Bernstein**. "A Spectral Condition for Feature Learning." (2023). [https://arxiv.org/abs/2310.17813](https://arxiv.org/abs/2310.17813)

6. **Jeremy Bernstein**. "Modular Manifolds." Thinking Machines Lab (2025). [https://thinkingmachines.ai/blog/modular-manifolds/](https://thinkingmachines.ai/blog/modular-manifolds/)

7. **Jianlin Su**. "Muon + Stiefel." Scientific Spaces (2025). [https://kexue.fm/archives/11221](https://kexue.fm/archives/11221)

8. **Jeremy Bernstein**. "Orthogonal manifold." Modula Systems Docs (2025). [https://docs.modula.systems/algorithms/manifold/orthogonal/](https://docs.modula.systems/algorithms/manifold/orthogonal/)

9. **Franz Louis Cesista**. "Heuristic Solutions for Steepest Descent on the Stiefel Manifold." (2025). [https://leloykun.github.io/ponder/steepest-descent-stiefel/](https://leloykun.github.io/ponder/steepest-descent-stiefel/)

10. **Jianlin Su**. "Thinking about Spectral Norm Gradient and Spectral Weight Decay." Scientific Spaces (2024). [https://kexue.fm/archives/10648](https://kexue.fm/archives/10648)

11. **Lizhang Chen, Jonathan Li, Qiang Liu**. "Muon Optimizes Under Spectral Norm Constraints." (2025). [https://arxiv.org/abs/2506.15054](https://arxiv.org/abs/2506.15054)

12. **Shikai Qiu, Zixi Chen, et al.** "Hyperparameter Transfer Enables Consistent Gains of Matrix-Preconditioned Optimizers Across Scale." (2025). [https://arxiv.org/abs/2512.05620](https://arxiv.org/abs/2512.05620)

13. **Zhiyuan Li, Kaifeng Lyu, Sanjeev Arora**. "Reconciling Modern Deep Learning with Traditional Optimization Analyses: The Intrinsic Learning Rate." NeurIPS (2020). [https://arxiv.org/abs/2010.02916](https://arxiv.org/abs/2010.02916)

14. **Ilya Loshchilov, Frank Hutter**. "Decoupled Weight Decay Regularization (AdamW)." (2019). [https://arxiv.org/abs/1711.05101](https://arxiv.org/abs/1711.05101)

15. **Atli Kosson, Bettina Messmer, Martin Jaggi**. "Rotational Equilibrium: How Weight Decay Balances Learning Across Neural Networks." (2024). [https://arxiv.org/abs/2305.17212](https://arxiv.org/abs/2305.17212)

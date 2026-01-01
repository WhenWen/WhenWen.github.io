---
layout: post
title: "From Weight Decay to Hyperball Optimization (Part 2): Weight decay theory deep dive"
description: "Part 2 of a two-part series: why weight decay sets effective step size in scale-invariant nets."
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
read_time: 30
---

> **Authors:** [Kaiyue Wen](https://whenwen.github.io/), [Xingyu Dang](https://dangxingyu.github.io/), [Kaifeng Lyu](https://kaifeng.ac/), [Tengyu Ma](https://ai.stanford.edu/~tengyuma/), [Percy Liang](https://cs.stanford.edu/~pliang/)

---

**Series navigation**

- **Part 1 (optimizer):** [Hyperball optimizer](/blog/2025/weight-decay-hyperball-part-1/)
- **Part 2 (this page):** Weight decay theory deep dive

---

Weight decay is a standard component of training, yet its role in modern deep learning is often misunderstood. In this post, we will show how recent deep learning research reveals that for scale-invariant models (like Transformers), weight decay does not control capacity. Instead, it controls the **effective step size** [1][2][3][4][5]. This theory is what motivates the design of the **Hyperball** optimizer in Part 1 of this series.

**Key Takeaways:**

1. **Debunk** the classical "capacity control" view.

![Illustrative figure showing weight decay concepts](/wd_blog/assets/images/fig0.png)

2. **Derive** the modern view: weight decay regulates weight norm, which then controls the effective update size. This can lead to unexpected phenomena. For example, gradient norms may increase as loss decreases [6]!

> **📊 Interactive Demo:** *W&B Metrics Plot*
> 
> `[Interactive visualization available in the HTML version]`
> 
> See the [interactive article](/wd_blog/public/weight-decay-part-2.html) for the full demo.

---

## 1. The Paradox of Weight Decay

Standard weight decay updates parameters $W$ by:

$$
W_{t+1} = (1 - \eta_t \lambda) W_t - \eta_t u_t
$$

where $\eta_t$ is the learning rate, $\lambda$ is the decay coefficient, and $u_t$ is the update direction given by the base optimizer. This is equivalent to minimizing $L(W) + \frac{\lambda}{2}\|W\|_F^2$ for SGD.

> **Classical View:** This penalty keeps weights small, limiting model capacity and preventing overfitting.
>
> **Modern Reality:** Most weight matrices in modern architectures (Transformers, ResNets with BatchNorm/LayerNorm) are **scale-invariant** [1][3]. Multiplying weights by a constant $c$ does not change the output or the loss:
>
> $$
> L(cW) = L(W), \quad \forall c > 0
> $$

> **📊 Interactive Demo:** *Scale Invariance Demo*
> 
> `[Interactive visualization available in the HTML version]`
> 
> See the [interactive article](/wd_blog/public/weight-decay-part-2.html) for the full demo.

So how scale-invariant are modern architectures, really? Let's take a look at the Transformer architecture.

> **📊 Interactive Demo:** *Transformer Demo*
> 
> `[Interactive visualization available in the HTML version]`
> 
> See the [interactive article](/wd_blog/public/weight-decay-part-2.html) for the full demo.

If the neural network function and hence the loss is unchanged by the scale of $W$, penalizing $\|W\|_F$ cannot constrain capacity. Yet, people continue to use weight decay. Why?

---

## 2. The Mechanism: How Weight Decay Sets the Effective Step Size

For scale-invariant losses $L(cW)=L(W)$, optimization depends only on the **direction** of the weights

$$
\hat{W} = W / \|W\|_F.
$$

This direction-only view follows the intrinsic learning-rate analysis of [4]. 

Define the **effective step size** $\eta_{\text{eff}}$ as the magnitude of the change in the weight direction:

$$
\eta_{\text{eff}} := \|\hat{W}_{t+1} - \hat{W}_t\|.
$$

![Illustrative figure showing weight decay concepts](/wd_blog/assets/images/fig1_output.png)

The key mechanism is:

1. The **weight norm** $\|W\|_F$ determines how large a step we take in direction space.
2. The **weight decay** coefficient $\lambda$ determines the equilibrium value of $\|W\|_F$.

Together these imply that $\lambda$ directly sets the effective step size. This calculation has been explored in detail in [4][8]. For readers who can read Chinese, excellent explanations have been provided in JianLin Su's blogs [9][10][11].

Before we dive into the calculations, let's showcase how current theory aligns with an interactive simulation. The demo below trains a simple normalized linear model using AdamW / Muon with gradient noise. Watch how the **empirical measurements** (solid lines) closely track the **theoretical predictions** (dashed lines). You can adjust the hyperparameters in real-time to see how $\eta$, $\lambda$, and $\beta_1$ affect the equilibrium behavior.

> **📊 Interactive Demo:** *AdamW Simulation Demo*
> 
> `[Interactive visualization available in the HTML version]`
> 
> See the [interactive article](/wd_blog/public/weight-decay-part-2.html) for the full demo.

The expressions below summarize the steady-state behavior of the three optimizers (AdamW, Muon, Moonlight) in the noise-dominated regime discussed above—this is the final outcome, so feel free to skip the following proof and jump straight to Section 2.6 if you are not in the mood for detailed calculations!

| Quantity | AdamW | Moonlight | Muon |
|----------|-------|-----------|------|
| $\|W_t\|_F$ | $\eta \sqrt{\frac{1-\beta_1}{1+\beta_1}}\sqrt{d_{\mathrm{in}}d_{\mathrm{out}}}\sqrt{\frac{1+\alpha\beta_1}{(1-\alpha^2)(1-\alpha\beta_1)}}$ | (same as AdamW) | $\eta \sqrt{d_{\mathrm{out}}}\sqrt{\frac{1+\alpha\beta_1}{(1-\alpha^2)(1-\alpha\beta_1)}}$ |
| $\|u_t\|_F$ | $\sqrt{\frac{1-\beta_1}{1+\beta_1}}\sqrt{d_{\mathrm{in}}d_{\mathrm{out}}}$ | (same as AdamW) | $\sqrt{d_{\mathrm{out}}}$ |
| $\cos(W_t,u_t)$ | $-\beta_1 \sqrt{\frac{1-\alpha^2}{(1+\alpha\beta_1)(1-\alpha\beta_1)}}$ | (same) | (same) |
| Effective step size $\eta_{\mathrm{eff}}$ | $\frac{1}{1+\alpha\beta_1}\sqrt{(1-\alpha^2)(1-\beta_1^2)} \approx \sqrt{2\eta\lambda \frac{1 - \beta_1}{1+\beta_1}}$ | (same) | (same) |

Here $\alpha = 1 - \eta\lambda$ and $d_{\mathrm{in}}, d_{\mathrm{out}}$ are the layer dimensions; Muon and Moonlight share the same correlation structure as AdamW, so only the update norm $U$ differs.

---

### 2.1 Basic Assumption: Noise-Dominated Training

Throughout this section we work in the **noise-dominated regime**: the stochasticity of the gradients is much larger than the signal. Concretely, for a single scalar gradient entry we assume

$$
g_t \in \mathbb{R},\quad
g_t \sim \mathcal N(0,\sigma^2)\ \text{i.i.d. over } t,
$$

and for a whole layer we treat the gradient as

$$
g_t \in \mathbb{R}^d,\quad
g_t \sim \mathcal N(0,\sigma^2 I_d).
$$

This assumption may look outrageous at first glance, because it ignores any structure in the loss and assumes there is "no signal." However, **for the specific quantities we care about** (stationary update norms, angles, equilibrium weight norms, effective step size), this is a good approximation when the noise level is much larger than the signal. It is the standard toy model: simple enough to be solvable, but rich enough to capture the scaling behavior with $\eta$ and $\lambda$. It is also interesting to note that a similar assumption has been used in [12] to study how batch size affects the effective step size.

---

### 2.2 Predicting Optimizer Update Norm 

Let $u_t$ be the **base optimizer update** *before* adding weight decay (e.g. the Adam part of AdamW). We show that, under the noise-dominated model, the update norm is approximately **constant over time**, depending only on the optimizer hyperparameters and the layer dimension.

For Muon, this is already guaranteed by design, as all of the singular values of Muon's update before scaling are $1$. Assuming the shape of $W$ is $d_{\mathrm{in}} \times d_{\mathrm{out}}$, we can derive the update norm as follows:

In the speedrun implementation of Muon, the update is scaled by $\max(\sqrt{\frac{d_{\mathrm{out}}}{d_{\mathrm{in}}}}, 1)$ [13], giving

$$
\|u_t\|_F = \sqrt{d_{\mathrm{out}}}.
$$

For the Moonlight implementation [14], the update is scaled by $0.2\sqrt{\max(d_{\mathrm{out}}, d_{\mathrm{in}})}$, giving

$$
\|u_t\|_F = 0.2\sqrt{d_{\mathrm{out}} d_{\mathrm{in}}}.
$$

For AdamW, focusing on a **single scalar coordinate** $\bar u_{t}$, we can write the (bias-corrected) update in the "infinite history" limit as

$$
\bar u_{t}
= \frac{m_t}{\sqrt{v_t}}, m_t = (1 - \beta_1)\sum_{i=0}^{\infty} \beta_1^i \bar g_{t-i}, v_t = (1 - \beta_2)\sum_{i=0}^{\infty} \beta_2^i \bar g_{t-i}^2
$$

where $\beta_1$ is the momentum coefficient, $\beta_2$ is the second-moment coefficient, and $\bar g_t$ is a scalar entry of the gradient, assumed i.i.d. $\mathcal N(0,\sigma^2)$ as in §2.1.

Under this assumption, the denominator is approximately constant:

<details>
<summary><strong>For a matrix $W_t$ with $d = d_{\mathrm{in}} d_{\mathrm{out}}$ number of parameters, $\mathbb{E}[\bar u_t^2] \approx d\frac{1-\beta_1}{1+\beta_1}$</strong></summary>

Assuming the denominator in AdamW converges to the variance of the gradient, we can compute the update norm by considering the variance of the momentum in the nominator in AdamW.

We first compute the expected value of the denominator:

$$
\begin{align*}
(1-\beta_2)\mathbb{E}[v_t] &= 
(1-\beta_2)\mathbb{E}[(1 - \beta_2)\sum_{i=0}^{\infty} \beta_2^i \bar g_{t-i}^2] \\
&=
(1-\beta_2)\mathbb{E}[\sum_{i=0}^{\infty} \beta_2^i \bar g_{t-i}^2] \\
&=
(1-\beta_2)\sigma^2 \sum_{i=0}^{\infty} \beta_2^i =
\sigma^2.
\end{align*}
$$

Plugging this approximation into the update gives

$$
\bar u_{t}
\approx
(1-\beta_1)
\frac{\sum_{i=0}^{\infty} \beta_1^i \bar g_{t-i}}{\sigma},
$$

Then

$$
\mathbb{E}\Big[
\Big(\frac{\sum_{i=0}^{\infty} \beta_1^i \bar g_{t-i}}{\sigma}\Big)^2
\Big]
=
\frac{1}{\sigma^2}\sum_{i=0}^{\infty} \beta_1^{2i} \mathbb{E}[\bar g_{t-i}^2]
=
\sum_{i=0}^{\infty} \beta_1^{2i}
=
\frac{1}{1-\beta_1^2}.
$$

Therefore

$$
\mathbb{E}[\bar u_t^2]
\approx
(1-\beta_1)^2 \cdot \frac{1}{1-\beta_1^2}
=
\frac{1-\beta_1}{1+\beta_1}.
$$

</details>

> **Property 1 (Approximately Constant Update Norm):** In the Gaussian noise model, common base optimizers like Adam and Muon have a **time-independent RMS**, so we can treat
>
> $$
> \|u_t\|_F \approx U
> $$
>
> as a **constant per layer**. 
>
> | Optimizer | Update Norm $\|u_t\|_F$ |
> |-----------|------------------------|
> | AdamW | $\sqrt{\frac{1-\beta_1}{1+\beta_1}} \sqrt{d_{\mathrm{in}} \times d_{\mathrm{out}}}$ |
> | Muon | $\sqrt{d_{\mathrm{out}}}$ |
> | Moonlight | $0.2\sqrt{d_{\mathrm{out}} d_{\mathrm{in}}}$ |
>
> Here $d_{\mathrm{in}}$ is the input dimension and $d_{\mathrm{out}}$ is the output dimension of the linear layer and $\beta_1$ is the momentum parameter.

---

### 2.3 Predicting the Correlation Between Update and Weight

We now turn to the relationship between the update vector $u_t$ and the weight vector $W_t$. A crucial observation is that they exhibit a **stable correlation**.

We quantify this via the projection coefficient $\gamma_t$:

$$
\gamma_t
:=
\frac{\langle u_t, W_t \rangle}{\|u_t\|_F^2}
\approx
\text{const}.
$$

**Intuition:** Why are they correlated?
1. **Weights accumulate history:** The weight vector $W_t$ is an exponentially weighted sum of **past updates** ($u_{t-1}, u_{t-2}, \dots$).
2. **Momentum creates memory:** Due to momentum, the current update $u_t$ is not independent of the past; it is strongly correlated with **recent past updates** ($u_{t-1}, u_{t-2}, \dots$).
3. **Correlation is inevitable:** Since $W_t$ is built from vectors that $u_t$ is correlated with, the projection of $W_t$ onto $u_t$ is non-zero and stable relative to the update scale.

Mathematically, with decoupled weight decay $W_{t+1} = \alpha W_t - \eta u_t$ (where $\alpha = 1-\eta\lambda$), we can write $W_t$ as a sum:

$$
W_t
=
-\eta\sum_{k=1}^{t} \alpha^{k-1} u_{t-k}.
$$

The inner product $\langle u_t, W_t \rangle$ becomes a weighted sum of auto-correlations $\langle u_t, u_{t-k} \rangle$. In a stationary noise regime, these auto-correlations are stable and scale with $U^2$, so $\gamma_t$ converges to a fixed value determined by $(\eta,\lambda,\beta_1)$.

<details>
<summary><strong>In the same noise-dominated regime, the correlation between $u_t$ and $W_t$ converges to $\gamma_t \approx -\frac{\eta\beta_1}{1-\alpha\beta_1}$</strong></summary>

Conceptually, the calculation is the same kind of "Gaussian covariance algebra" as in §2.2: we write down a linear recurrence for $u_t$ and $W_t$, assume Gaussian noise for the gradients, and solve for the stationary covariance.

Start from SGD with momentum (for a single coordinate) and decoupled weight decay:

$$
\begin{aligned}
u_t &= (1-\beta_1)\sum_{i=0}^t \beta_1^{t-i} g_i, \\
W_{t+1} &= \alpha W_t - \eta u_t,
\end{aligned}
$$

with $g_t \sim \mathcal N(0,\sigma^2 I_d)$ i.i.d. as before.

In high dimension we have

$$
\langle g_i, g_j \rangle \approx \sigma^2 d\,\delta_{ij},
$$

so for any two times $t,t'$,

$$
\begin{aligned}
\langle u_t, u_{t'} \rangle
&=
\Big\langle
(1-\beta_1)\sum_{i=0}^t \beta_1^{t-i} g_i,\,
(1-\beta_1)\sum_{j=0}^{t'} \beta_1^{t'-j} g_j
\Big\rangle \\
&=
\frac{(1-\beta_1)^2}{(1-\beta_1^{t+1})(1-\beta_1^{t'+1})}
\sum_{i=0}^t \sum_{j=0}^{t'} \beta_1^{t+t'-i-j}
\langle g_i, g_j \rangle \\
&\approx
\frac{(1-\beta_1)^2}{(1-\beta_1^{t+1})(1-\beta_1^{t'+1})}
\sum_{i=0}^{\min(t,t')} \beta_1^{t+t'-2i} \sigma^2 d \\
&=
\frac{(1-\beta_1)^2}{(1-\beta_1^{t+1})(1-\beta_1^{t'+1})}
\sigma^2 d\,
\beta_1^{t+t'-2\min(t,t')}
\frac{1-\beta_1^{2\min(t,t')+2}}{1-\beta_1^2} \\   
&=
\frac{(1-\beta_1)(1-\beta_1^{2\min(t,t')+2})\beta_1^{t+t'-2\min(t,t')}}
{(1-\beta_1^{t+1})(1-\beta_1^{t'+1})}
\sigma^2 d.
\end{aligned}
$$

This shows that the correlation between $u_t$ and $u_{t'}$ **decays geometrically** with $|t-t'|$, controlled by $\beta_1$. Since

$$
W_t = -\eta\sum_{k=0}^{t-1} \alpha^{t-1-k} u_k,
$$

we can write

$$
\langle W_t, u_t \rangle
=
-\eta \sum_{k=0}^{t-1} \alpha^{t-1-k} \langle u_k, u_t \rangle.
$$

Plugging in $\langle u_k, u_t \rangle \approx \|u_t\|_F^2 \beta_1^{t-k}$ (valid for large $t$):

$$
\begin{aligned}
\langle W_t, u_t \rangle
&\approx
-\eta \sum_{k=0}^{t-1} \alpha^{t-1-k} \left( \|u_t\|_F^2 \beta_1^{t-k} \right) \\
&=
-\eta \|u_t\|_F^2 \beta_1 \sum_{k=0}^{t-1} (\alpha\beta_1)^{t-1-k}.
\end{aligned}
$$

As $t\to\infty$, the sum becomes a geometric series $\sum_{j=0}^\infty (\alpha\beta_1)^j = \frac{1}{1-\alpha\beta_1}$. Thus:

$$
\langle W_t, u_t \rangle
\approx
-\eta \|u_t\|_F^2 \frac{\beta_1}{1-\alpha\beta_1}.
$$

Dividing by $\|u_t\|_F^2$ gives the constant projection coefficient:

$$
\gamma 
= \frac{\langle W_t, u_t \rangle}{\|u_t\|_F^2}
\approx
-\frac{\eta\beta_1}{1-\alpha\beta_1}.
$$

</details>

In practice, for the regimes of interest this projection is stable.

> Because $W_t$ accumulates past updates and momentum ensures $u_t$ correlates with those same past updates, the **projection of $W_t$ onto $u_t$** stabilizes. Specifically, 
>
> $$
> \langle W_t, u_t \rangle \approx -\frac{\eta\beta_1}{1-\alpha\beta_1} \|u_t\|_F^2.
> $$
>
> This projection term is the same across AdamW and Muon. 

---

### 2.4 Solving the Equilibrium Weight Norm

We are now ready to close the loop. The dynamics of the weight norm are driven by a tension between two forces: **weight decay**, which shrinks the weights, and **optimizer updates**, which drive the weights away from zero.

Combining our findings from the previous sections, we can reduce the high-dimensional vector dynamics to a simple **scalar recursion**.

Recall our two key properties:
1. **Constant Update Size (§2.2):** The update norm is constant, $\|u_t\|_F \approx U$.
2. **Stable Projection (§2.3):** The alignment between weights and updates is fixed, $\langle W_t, u_t \rangle \approx \gamma U^2$.

With decoupled weight decay, the weight evolves as $W_{t+1} = \alpha W_t - \eta u_t$. Squaring this equation gives the evolution of the norm $r_t := \|W_t\|_F$:

$$
\begin{aligned}
r_{t+1}^2 
&= \|\alpha W_t - \eta u_t\|_F^2 \\
&= \alpha^2 \|W_t\|_F^2 + \eta^2 \|u_t\|_F^2 - 2\alpha\eta \langle W_t, u_t \rangle.
\end{aligned}
$$

Substituting our approximations for $\|u_t\|_F$ and $\langle W_t, u_t \rangle$, we obtain a closed 1-D system:

$$
r_{t+1}^2 \approx \alpha^2 r_t^2 + \underbrace{(\eta^2 - 2\alpha\eta \gamma) U^2}_{\text{effective norm increase}}. \quad (\star)
$$

> **Interpretation:** The term $(\eta^2 - 2\alpha\eta \gamma) U^2$ represents the *effective* amount of norm increase at each step. It is less than the raw update norm times learning rate (i.e. $\eta^2 U^2$) because the update $u_t$ is **anti-correlated** with $W_t$ (due to momentum), which dampens the expansion.

#### The Steady State

At equilibrium, the expected norm stabilizes ($r_{t+1} \approx r_t \approx r_\star$). Solving $(\star)$ for the stationary value $r_\star$:

$$
r_\star^2 (1 - \alpha^2) \approx (\eta^2 - 2\alpha\eta\gamma) U^2 
\implies 
r_\star \approx U \sqrt{\frac{\eta^2 - 2\alpha\eta\gamma}{1-\alpha^2}}.
$$

Finally, we plug in the specific projection coefficient $\gamma \approx -\frac{\eta\beta_1}{1-\alpha\beta_1}$ derived in §2.3. After simplifying the algebra, we arrive at the **exact equilibrium norm** for a scale-invariant layer trained with AdamW/Muon:

$$
\boxed{
\|W_\infty\|_F 
\approx 
\eta U \sqrt{\frac{1+\alpha\beta_1}{(1-\alpha^2)(1-\alpha\beta_1)}}
}
$$

This formula is powerful because it depends *only* on the hyperparameters ($\eta, \lambda, \beta_1$) and the layer geometry (through $U$). It requires no empirical fitting.

> The equilibrium weight norm with respect to the expected update norm $U$ is fully determined by the optimizer settings.
>
> $$
> \|W_\infty\|_F 
> \approx 
> \eta U \sqrt{\frac{1+\alpha\beta_1}{(1-\alpha^2)(1-\alpha\beta_1)}}
> $$

---

### 2.5 Solving the Effective Step Size

We finally translate the norm dynamics into an **effective step size**, defined as the magnitude of the change in the *direction* of the weights:

$$
\eta_{\mathrm{eff},t}
:=
\|\hat{W}_{t+1} - \hat{W}_t\|_F.
$$

This measures how fast the model traverses the function landscape, independent of the weight scale.

#### Deriving the Step Size

The step size is determined by the projection of the update onto the tangent space of the unit sphere.

<details>
<summary><strong>Given that we know the relative norm ratio between the update and the weight and their correlation, we can calculate the effective step size.</strong></summary>

The effective step size is given by:

$$
\begin{align*}
\eta_{\mathrm{eff},t}
&= \frac{1}{1 + \alpha \beta_1} \sqrt{(1 - \alpha^2)(1 - \beta_1^2)}
\end{align*}
$$

Note that $\alpha = 1 - \eta\lambda$ and $\beta_1$ is the momentum coefficient. If we assume that $\eta\lambda$ is small, then we can approximately get $\eta_{\mathrm{eff},t} \approx \sqrt{2\eta\lambda \frac{1 - \beta_1}{1 + \beta_1}}$.

**Step 1: Geometric Approximation**

The effective step size is the magnitude of the update projected onto the tangent space of the unit sphere. Let $P_{w^\perp} = I - \hat{w}\hat{w}^\top$ (noted here we view weight as a vector instead of a matrix).

$$
\eta_{\mathrm{eff},t} \approx \frac{\eta}{\|w_t\|_F} \|P_{w_t^\perp} u_t\|_F.
$$

**Step 2: Tangential Component**

Let $k = \|u_t\|_F / \|W_t\|_F$ and using the projection coefficient $\gamma$ from Section 2.3, we can express the tangential norm as:

$$
\begin{align*}
\|P_{W_t^\perp} u_t\|_F^2 &= \|u_t - \frac{\langle u_t, W_t \rangle}{\|W_t\|_F^2} W_t\|_F^2\\
&= \| u_t \|^2 - \frac{(\langle u_t, W_t \rangle)^2}{\|W_t\|_F^2} \\
&\approx \| u_t \|^2 (1 - \gamma^2 k^2)
\end{align*}
$$

Therefore, the effective step size is:

$$
\eta_{\mathrm{eff},t} = \eta k \sqrt{1 - \gamma^2 k^2}
$$

**Step 3: Steady-State Substitution**

From previous section, we have the equilibrium condition:

$$
\begin{align*}
k^2 &= \frac{(1 - \alpha^2)(1 - \alpha\beta_1)}{\eta^2 (1 + \alpha\beta_1)} \\
\gamma^2 &= \frac{\eta^2\beta_1^2}{(1-\alpha\beta_1)^2}
\end{align*}
$$

Therefore we can simplify the expression for the effective step size to 

$$
\begin{align*}
\eta_{\mathrm{eff},t}
& =
\sqrt{\frac{(1 - \alpha^2)(1 - \alpha \beta_1)}{1 + \alpha \beta_1}}
\sqrt{
1 -
\frac{\beta_1^2 (1 - \alpha^2)}
{(1 + \alpha \beta_1)(1 - \alpha \beta_1)}
} \\
&= \frac{1}{1 + \alpha \beta_1} 
\sqrt{
(1 - \alpha^2) \big((1 - \alpha^2 \beta_1^2) - \beta_1^2 (1 - \alpha^2)\big)
}  \\
&= \frac{1}{1 + \alpha \beta_1} \sqrt{(1 - \alpha^2)(1 - \beta_1^2)}.
\end{align*}
$$

This concludes our calculation and show that the effective step size is determined by the product of the learning rate and the weight decay.

</details>

> In steady state, the effective step size on the unit sphere scales like
>
> $$
> \eta_{\mathrm{eff},t} \approx \frac{1}{1 + \alpha \beta_1} \sqrt{{(1 - \alpha^2)(1 - \beta_1^2)}} \approx \sqrt{2\eta\lambda \frac{1 - \beta_1}{1 + \beta_1}}.
> $$
>
> Weight decay $\lambda$ and learning rate $\eta$ together define a **hidden effective step size**: tuning $\eta\lambda$ directly controls how aggressively the model moves in direction space [4][8].

---

### 2.6 One Last Thing: How Gradient Norms Scale

Finally, we explain why **gradient norms tend to grow** toward the end of training when the learning rate decays, even as the loss keeps decreasing. This calculation was shown previously.

From §2.4, in the steady-state regime we have (up to a layer-dependent constant $C(\alpha,\beta)$)

$$
\|W_t\|_F
\approx
C(\alpha,\beta)\sqrt{\frac{\eta_t}{\lambda}}.
$$

For a **scale-invariant** layer, the loss satisfies

$$
L(cW) = L(W)\quad\forall c>0,
$$

which implies the gradient rescales inversely:

$$
\nabla_W L(cW) = \frac{1}{c} \nabla_W L(W).
$$

<details>
<summary><strong>Scale invariance implies inverse gradient scaling</strong></summary>

A larger weight norm means that a fixed norm update makes less change to the direction of the feature.

For scale-invariant loss, by definition

$$
L(cW + c\epsilon) - L(cW)
=
L(W + \epsilon) - L(W)
$$

for any perturbation $\epsilon$. Differentiating w.r.t. $\epsilon$ at $\epsilon=0$ gives

$$
\langle \nabla_W L(cW), c\epsilon \rangle
=
\langle \nabla_W L(W), \epsilon \rangle
\quad\forall \epsilon,
$$

so

$$
\nabla_W L(cW) = \frac{1}{c}\,\nabla_W L(W).
$$

Thus, multiplying weights by $c$ divides the gradient norm by $c$.

</details>

Therefore, in the scale-invariant regime

$$
\|G_t\|_F
:=
\|\nabla_W L(W_t)\|_F
\propto
\frac{1}{\|W_t\|_F}
\propto
\sqrt{\frac{\lambda}{\eta_t}}.
$$

So as the **learning rate decays** during training, the **equilibrium weight norm shrinks**, and the **gradient norm grows**:

* loss can keep going down,
* but $\|G_t\|_F$ naturally increases as $\eta_t$ decreases.

This looks like "gradient explosion" in the logs, but in this model it is simply the **expected behavior of scale invariance + weight decay + LR decay**. This calculation was previously shown in [4][6].

> **Key point (2.6):** In scale-invariant layers with weight decay, the steady-state weight norm scales like
>
> $$
> \|W_t\|_F \propto \sqrt{\eta_t/\lambda},
> $$
>
> so the gradient norm scales like
>
> $$
> \|G_t\|_F \propto \sqrt{\lambda/\eta_t}.
> $$
>
> When the learning rate decays, the **equilibrium radius shrinks and the gradient norm rises**, explaining the empirically observed increase in gradient norms late in training.

---

## 3. Explaining Empirical Phenomena

The classic saying goes "all models are wrong, but some are useful". In deep learning theory, whether a theory is useful should be judged by two perspectives:

1. Can it predict empirical phenomena? (This section)
2. Can it motivate algorithm that works better? (Part 1)

Most of the empirical figures in this section are from the optimizer sweeps from our previous study of pretraining optimizers [15].

### Phenomenon 1: Weight norm tracks learning rate warmup and decay throughout training.

**Explanation:** As we have derived in Section 2.3, the equilibrium weight norm is determined by learning rate $\eta$ and weight decay $\lambda$, as $\|W_{\infty}\|_F \propto \sqrt{\frac{\eta}{2\lambda}} U$. 

> **📊 Interactive Demo:** *W&B QKV Norms Plot*
> 
> `[Interactive visualization available in the HTML version]`
> 
> See the [interactive article](/wd_blog/public/weight-decay-part-2.html) for the full demo.

### Phenomenon 2: Gradient norm increases through training.

**Explanation:** As we have derived in Section 2.5, the gradient norm is determined by the weight norm, as $\|G_t\|_F \propto \frac{1}{\|W_t\|_F} \propto \sqrt{\lambda/\eta}$. Naturally, as the weight norm decreases, the gradient norm increases.

> **📊 Interactive Demo:** *W&B QKV Gradient Norms Plot*
> 
> `[Interactive visualization available in the HTML version]`
> 
> See the [interactive article](/wd_blog/public/weight-decay-part-2.html) for the full demo.

### Phenomenon 3: When $\eta \lambda$ is fixed, the model trained with AdamW converges to essentially the same loss. At the same time, the weight norm of each weight matrix is (roughly) proportional to the learning rate.

**Explanation:** As we have derived in Section 2.5, the effective step size is determined by learning rate $\eta$ and weight decay $\lambda$, as $\eta_{\mathrm{eff}} \propto \sqrt{\eta\lambda}$. Further, as derived in Section 2.4, the equilibrium weight norm $\|W_{\infty}\|_F \propto \sqrt{\frac{\eta}{\lambda}} U$ is proportional to the learning rate when $\eta \lambda$ is fixed [13].

> **📊 Interactive Demo:** *W&B Layer 9 Comparison Plot*
> 
> `[Interactive visualization available in the HTML version]`
> 
> See the [interactive article](/wd_blog/public/weight-decay-part-2.html) for the full demo.

### Phenomenon 4: Despite sharing the same learning-rate schedule, the run with weight decay starts with a higher loss but ultimately converges to a strictly lower loss than the run without weight decay.

**Explanation:** Although these two runs use the same nominal learning-rate warmup and decay schedule, weight decay changes the weight norms and therefore induces a substantially different effective step size over training. Empirically (and in the theory in Section 2), we find that training with weight decay yields a larger effective step size throughout training than training without weight decay. In the *river valley landscape* picture [16], the loss decomposes into a "river" component, capturing progress along a relatively flat direction where long-term optimization happens, and a "hill" component, capturing excursions in steep directions caused by stochastic gradients.

![River valley landscape schematic](/wd_blog/assets/images/function2.png)
*Visual intuition for the river valley landscape conjecture.*

A larger effective step size amplifies these hill-direction oscillations, which raises the observed loss early in training, but it also accelerates motion along the river. When the learning rate decays, the oscillations in the hill directions shrink and the iterate settles closer to the riverbed, revealing the additional progress that has already been made along the river. In our setting, the run with weight decay therefore starts with a higher loss but ultimately reaches a lower loss, because its larger effective step size allows it to move faster down the river before the decay phase suppresses the oscillations.

> **📊 Interactive Demo:** *W&B Analysis Plot*
> 
> `[Interactive visualization available in the HTML version]`
> 
> See the [interactive article](/wd_blog/public/weight-decay-part-2.html) for the full demo.

### Phenomenon 5: Contrary to the original MuP prediction, hyperparameter transfer is not sensitive to weight scale at initialization but it is sensitive to how weight decay is scaled

**Explanation:** Weight decay has emerged as the key driver of hyperparameter transfer in recent work [17][18][19][20], overshadowing MuP initialization [21]. This isn't surprising: the weight norm rapidly settles into the equilibrium predicted by theory, a value set solely by the learning rate and weight decay and it does **not** depend on the initial scale. Because Tensor Program theory assumes hyperparameter transfer is enabled by the weight norm and the properly rescaled update norm staying stable across network width and depth, weight decay naturally becomes more influential than initialization.

![Comparison of learning rate transfer strategies with and without weight decay](/wd_blog/assets/images/llama_ind_vs_std_vs_nowd_teaser.png)
*Figure 1 in [17].*

---

## Acknowledgments

The authors would like to thank [Songlin Yang](https://sustcsonglin.github.io/), [Zihan Qiu](https://www.linkedin.com/in/zihan-qiu-33a172249/), and [Liliang Ren](https://renll.github.io/) for motivating this blog post into existence. To some extent, this work is a proof to show that it is possible to remove weight decay altogether by designing the optimizer to explicitly control weight norms. The authors would also like to thank [William Held](https://williamheld.com/), [David Hall](http://dlwh.org/), [Suhas Kotha](https://kothasuhas.github.io/), [Tatsunori Hashimoto](https://thashim.github.io/), [Jason Lee](https://jasondlee88.github.io/), [Zhiyuan Li](https://zhiyuanli.ttic.edu/), [Lijie Chen](https://chen-lijie.github.io/), [Huaqing Zhang](https://scholar.google.com/citations?user=_E9tcTkAAAAJ), [Jiacheng You](https://github.com/YouJiacheng), [Jeremy Bernstein](https://jeremybernste.in/) and [Samuel Schoenholz](https://www.linkedin.com/in/samuel-schoenholz-379830a0/) for helpful discussions.

---

## Citations

If this work is helpful to you, please consider citing:

```bibtex
@online{wen2025hyperball,
    title        = {Fantastic Pretraining Optimizers and Where to Find Them II: From Weight Decay to Hyperball Optimization},
    author       = {Wen, Kaiyue and Dang, Xingyu and Lyu, Kaifeng and Ma, Tengyu and Liang, Percy},
    year         = {2025},
    month        = {11},
    day          = {30},
    url          = {https://whenwen.github.io/wd_blog/public/hyperball-part-1.html},
    urldate      = {2025-12-15},
}
```

---

## References

1. **Twan van Laarhoven**. "L2 Regularization versus Batch and Weight Normalization." arXiv (2017). [https://arxiv.org/abs/1706.05350](https://arxiv.org/abs/1706.05350)

2. **Guodong Zhang, Chaoqi Wang, Bowen Xu, Roger Grosse**. "Three Mechanisms of Weight Decay Regularization." ICLR (2019). [https://openreview.net/forum?id=B1lz-3Rct7](https://openreview.net/forum?id=B1lz-3Rct7)

3. **Elad Hoffer, Ron Banner, Itay Golan, Daniel Soudry**. "Norm matters: efficient and accurate normalization schemes in deep networks." (2018). [https://arxiv.org/abs/1803.01814](https://arxiv.org/abs/1803.01814)

4. **Zhiyuan Li, Kaifeng Lyu, Sanjeev Arora**. "Reconciling Modern Deep Learning with Traditional Optimization Analyses: The Intrinsic Learning Rate." NeurIPS (2020). [https://arxiv.org/abs/2010.02916](https://arxiv.org/abs/2010.02916)

5. **Francesco D'Angelo, Maksym Andriushchenko, Aditya Varre, Nicolas Flammarion**. "Why Do We Need Weight Decay in Modern Deep Learning?" (2023). [https://arxiv.org/abs/2310.04415](https://arxiv.org/abs/2310.04415)

6. **Aaron Defazio**. "Why Gradients Rapidly Increase Near the End of Training." (2025). [https://arxiv.org/abs/2506.02285](https://arxiv.org/abs/2506.02285)

7. **Ilya Loshchilov, Frank Hutter**. "Decoupled Weight Decay Regularization (AdamW)." (2019). [https://arxiv.org/abs/1711.05101](https://arxiv.org/abs/1711.05101)

8. **Atli Kosson, Bettina Messmer, Martin Jaggi**. "Rotational Equilibrium: How Weight Decay Balances Learning Across Neural Networks." (2024). [https://arxiv.org/abs/2305.17212](https://arxiv.org/abs/2305.17212)

9. **Jianlin Su**. "AdamW Weight RMS Asymptotics (Part I)." (2025). [https://kexue.fm/archives/11307](https://kexue.fm/archives/11307)

10. **Jianlin Su**. "Why Adam's Update RMS Is 0.2?" (2025). [https://kexue.fm/archives/11267](https://kexue.fm/archives/11267)

11. **Jianlin Su**. "AdamW Weight RMS Asymptotics (Part II)." (2025). [https://kexue.fm/archives/11404](https://kexue.fm/archives/11404)

12. **Sadhika Malladi, Kaifeng Lyu, Abhishek Panigrahi, Sanjeev Arora**. "On the SDEs and Scaling Rules for Adaptive Gradient Algorithms." (2022). [https://arxiv.org/abs/2205.10287](https://arxiv.org/abs/2205.10287)

13. **Keller Jordan**. "Muon: An optimizer for hidden layers in neural networks." (2023). [https://kellerjordan.github.io/posts/muon/](https://kellerjordan.github.io/posts/muon/)

14. **Jingyuan Liu, Jianlin Su, et al.** "Muon is Scalable for LLM Training." (2025). [https://arxiv.org/abs/2502.16982](https://arxiv.org/abs/2502.16982)

15. **Kaiyue Wen, David Hall, Tengyu Ma, Percy Liang**. "Fantastic Pretraining Optimizers and Where to Find Them." (2025). [https://arxiv.org/abs/2509.02046](https://arxiv.org/abs/2509.02046)

16. **Kaiyue Wen, Zhiyuan Li, Jason Wang, David Hall, Percy Liang, Tengyu Ma**. "Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective." (2024). [https://arxiv.org/abs/2410.05192](https://arxiv.org/abs/2410.05192)

17. **Atli Kosson, Jeremy Welborn, Yang Liu, Martin Jaggi, Xi Chen**. "Weight Decay may matter more than μP for Learning Rate Transfer in Practice." (2025). [https://arxiv.org/abs/2510.19093](https://arxiv.org/abs/2510.19093)

18. **Charlie Blake, et al.** "u-μP: The Unit-Scaled Maximal Update Parametrization." (2024). [https://arxiv.org/abs/2407.17465](https://arxiv.org/abs/2407.17465)

19. **Zhiyuan Fan, Yifeng Liu, Qingyue Zhao, Angela Yuan, Quanquan Gu**. "Robust Layerwise Scaling Rules by Proper Weight Decay Tuning." (2025). [https://arxiv.org/abs/2510.15262](https://arxiv.org/abs/2510.15262)

20. **Xi Wang, Laurence Aitchison**. "How to set AdamW's weight decay as you scale model and dataset size." (2024). [https://arxiv.org/abs/2405.13698](https://arxiv.org/abs/2405.13698)

21. **Greg Yang, Edward J. Hu, et al.** "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer." (2021). [https://arxiv.org/abs/2203.03466](https://arxiv.org/abs/2203.03466)

22. **Tim Salimans, Diederik P. Kingma**. "Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks." (2016). [https://arxiv.org/abs/1602.07868](https://arxiv.org/abs/1602.07868)

23. **Tero Karras, et al.** "Analyzing and Improving the Training Dynamics of Diffusion Models." (2023). [https://arxiv.org/abs/2312.02696](https://arxiv.org/abs/2312.02696)

24. **Yonggan Fu, et al.** "Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models." (2025). [https://arxiv.org/abs/2511.18890](https://arxiv.org/abs/2511.18890)

25. **Greg Yang, James B. Simon, Jeremy Bernstein**. "A Spectral Condition for Feature Learning." (2023). [https://arxiv.org/abs/2310.17813](https://arxiv.org/abs/2310.17813)

26. **Jeremy Bernstein**. "Modular Manifolds." Thinking Machines Lab (2025). [https://thinkingmachines.ai/blog/modular-manifolds/](https://thinkingmachines.ai/blog/modular-manifolds/)

27. **Jianlin Su**. "Muon + Stiefel." Scientific Spaces (2025). [https://kexue.fm/archives/11221](https://kexue.fm/archives/11221)

28. **Jeremy Bernstein**. "Orthogonal manifold." Modula Systems Docs (2025). [https://docs.modula.systems/algorithms/manifold/orthogonal/](https://docs.modula.systems/algorithms/manifold/orthogonal/)

29. **Franz Louis Cesista**. "Heuristic Solutions for Steepest Descent on the Stiefel Manifold." (2025). [https://leloykun.github.io/ponder/steepest-descent-stiefel/](https://leloykun.github.io/ponder/steepest-descent-stiefel/)

30. **Jianlin Su**. "Thinking about Spectral Norm Gradient and Spectral Weight Decay." Scientific Spaces (2024). [https://kexue.fm/archives/10648](https://kexue.fm/archives/10648)

31. **Lizhang Chen, Jonathan Li, Qiang Liu**. "Muon Optimizes Under Spectral Norm Constraints." (2025). [https://arxiv.org/abs/2506.15054](https://arxiv.org/abs/2506.15054)

32. **Shikai Qiu, et al.** "Hyperparameter Transfer Enables Consistent Gains of Matrix-Preconditioned Optimizers Across Scale." (2025). [https://arxiv.org/abs/2512.05620](https://arxiv.org/abs/2512.05620)

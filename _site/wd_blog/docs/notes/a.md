In the rest of this post, we will show the following hidden relationship of weight decay in deep learning.



and show how to use norm-constrained training to get better performance.


1. Review the classical view of weight decay as L2 regularization.
2. Show how scale invariance breaks that view and derive a 'new' interpretation—weight decay influence the norm of $W_t$ to keep the *intrinsic learning rate* steady.
3. Show how the interpretation explain counterintuitive empirical phenomenon
4. Finally, demonstrate how to remove weight decay altogether by designing the optimizer to explicitly control weight norms.  
   1. This optimizer enable almost perfect hyperparameter transfer to 8$\times$ wider or 8$\times$ deeper model **without** manually changing the initialization  
   2. It speedup pretraining compared to varying-norm AdamW or Muon baseline by 10-20%


Weight decay has quietly accompanied every era of deep learning—from logistic regression to billion-parameter Transformers—yet its true purpose in modern training remains puzzling.

Formally, weight decay adds a simple multiplicative shrinkage to every parameter update:
$$
W_{t+1} = (1 - \eta_t \lambda) W_t - \eta_t u_t,
$$
where $u_t$ denotes the update direction provided by the base optimizer (e.g., Adam, Muon, or SGD *without* weight decay), $\eta_t$ is the learning rate, and $\lambda$ is the decay coefficient.
When the base optimizer is GD, $u_t = \nabla L(w)$ for some loss $L$, it corresponds to minimizing a regularized objective:
$$
L(W) + \tfrac{\lambda}{2}\|W\|_F^2.
$$
In [classical](#Section 2. Classical View of Weight Decay) machine learning, this term limits the model’s capacity by keeping weights small, and therefore improves the generalization.

However, this reasoning breaks down for modern architectures that are mostly **scale-invariant**. In such systems, scaling some weights by a constant $c$ leaves the loss unchanged:
$$
 L(cW) = L(W), \qquad \forall c > 0.
$$
[TBD: A figure showing what is scale invariance]


Deal to the wide-use of normalization, many of the weights in the current Transformers architecture is scale-invariant. If the model’s predictions do not depend on $\|W\|_F$, then penalizing $\|W\|_F$ can’t meaningfully constrain capacity.

Empirically, what happens if we turn off weight decay? Surprisingly,

> turning off weight decay often yields **lower loss early in training**, yet **worse loss at convergence**.


## Section 1. Why weight decay still matters (and why it’s confusing)





![image-20251019171210722](https://hackmd.io/_uploads/B1S1hD2Cxe.png)


[TBD:cap]


[TBD: gradient norm small -> increase]



## Section 2. Classical View of Weight Decay

In classical optimization, **weight decay** is equivalent to **L2 regularization**. For a model with parameters $W$, the regularized objective is
$$
 \tilde{L}(W) = L(W) + \frac{\lambda}{2}\|W\|_F^2,
$$


whose gradient is
$$
 \nabla_W \tilde{L}(W) = \nabla_W L(W) + \lambda W.
$$


An update step with learning rate $\eta_t$ gives
$$
W_{t+1} = W_t - \eta_t (\nabla_W L(W_t) + \lambda W_t)
 = (1 - \eta_t \lambda)W_t - \eta_t u_t,
$$
where $u_t$ is the SGD's update direction. This view interprets weight decay as a continuous *shrinkage* of the parameter vector toward the origin.

The classical justification for this shrinkage comes from **capacity control**: smaller weights yield smaller hypothesis complexity.


## Section 3. The Modern View — Weight Decay Controls Implicit Learning Rate Schedule

The classical justification for weight decay assumes that the loss $L(W)$ depends on the **absolute scale** of $W$. However, this assumption no longer holds for most modern neural architectures. Transformers and ResNet are built from **scale-invariant components**—LayerNorm and RMSNorm—such that multiplying all weights by a constant $c$ leaves the network’s output unchanged:
$$
L(cW) = L(W), \qquad \forall c > 0.
$$

The function class no longer expands with $\|W\|$. Consequently, bounding $\|W\|$ no longer limits capacity, and the generalization argument collapses.

### A concrete example: QK-Norm attention

Consider the attention mechanism with normalized queries and keys:
$$
\text{Attn}(Q, K, V) = \mathrm{softmax} \left(\frac{\mathrm{RMSNorm}(Q)\mathrm{RMSNorm}(K)^\top}{\sqrt{d}}\right) V,
$$
where
$$
\mathrm{RMSNorm}(x) = \frac{x}{\sqrt{\tfrac{1}{d}\|x\|_2^2 + \epsilon}}.
$$
If we scale all $W_Q$ and $W_K$ matrices by $c$, i.e.
$$
 Q = c W_Q , \qquad K = c W_K
$$
then both $\mathrm{RMSNorm}(Q)$ and $\mathrm{RMSNorm}(K)$ are unchanged—the scale cancels out. Consequently, the entire attention output (and therefore the loss)  remain invariant to the magnitude of $W_Q$ and $W_K$.   

### Weight Norm Controls the Effective Learning Rate

As weight norm are now disentangled from the forward process of the neural network, the training process should now be understood in terms of the **normalized weights** as they maintain functionality.
$$
\hat{W} = \mathrm{Norm}(W) = \frac{W}{\|W\|_F} \implies f(x; \hat W) = f(x; W)
$$
which capture the parameter *direction* that actually affects the loss, while $\|W\|_F$ merely acts as a scale factor disconnected from model behavior.

To analyze how the optimizer changes $\hat{W}$, consider one step of the update:
$$
W_{t+1} = W_t - \eta_t u_t.
$$


The normalized parameter after the step is
$$
\hat{W}_{t+1} = \mathrm{Norm}\big(W_t - \eta_t u_t\big) \approx 
 \hat{W}_t - \frac{\eta_t}{\|W_t\|_F} u_t + O(\eta_t^2)
$$
assuming $\eta_t$ is small enough so that $\|W_t\|_F$ changes slowly. Crucially, the step magnitude on $\hat{W}$ is scaled by $\eta_t / \|W_t\|_F$.  

We can define the **effective learning rate on the normalized parameters** as the norm of weight changes and this term is inversely proportional to the weight norm
$$
\eta_{eff,t} = \frac{\eta_t}{\|W_t\|_F} \|u_t\|_F.
$$



Then the corresponding update will be
$$
\hat W_{t + 1} \approx \mathrm{Norm}(\hat W_t - \eta_{eff,t} \mathrm{Norm}(u_t)).
$$

### Weight Decay Controls The Weight Norm

Interestingly, the weight norm when training with weight decay and modern optimizers like Adam and Muon can be easily estimated from the hyperparameters. To see this effect, we consider a simple optimizer  called Normalized GD:
$$
u_t = \frac{\nabla_{W} L(W_t)}{\| \nabla_{W} L(W_t) \|_F}.
$$
We can observe two important properties, which both hold approximately true on AdamW and Muon: 

**Property 1: Orthogonality between optimizer update and weight**

Because the loss is scale invariant, by Taylor Expansion [TBD: Explain the picture of this ]
$$
L(W) = L(W + \epsilon W) = L(W) + \langle \nabla_ W L(W), \epsilon W\rangle + O(\epsilon^2)
$$
Therefore, 
$$
\langle \nabla_W L, W \rangle = 0.
$$
This means that the gradient and the weight are always orthogonal when the architecture is scale invariant.

Therefore, for the normalized GD optimizer, we have that 
$$
\langle u_t, W_t \rangle = 0.
$$

[What is this angle actually?]


**Property 2: Constant norm optimizer update**

Here as we normalized the update, it always has Frobenius Norm of 1. For many of the modern optimizers, this approximately hold:

1. Adam

Adam can be viewed as a smooth version of SignGD $u_t = \mathrm{sign}(\nabla_W L(W_t))$ where $\mathrm{sign}(x)$ indicates the sign of the update. Therefore, it holds that
$$
\| u_{t,\mathrm{Adam}} \| \approx \sqrt{\mathrm{Number\ of \ Element\ in\ } W}
$$
In practice, they are typically a constant factor near $0.2$.

2. Muon

All of the singular value of Muon's update is $1$, therefore
$$
\|u_{t, \mathrm{Muon}} \| = \left(\sum_{\sigma_i \in \mathrm{singular\ value\ of\ u_{t,\mathrm{Muon}} }} \sigma_i^2 \right)^{1/2} = \sqrt{\mathrm{Smallest\ Dimension\ of\ }W}.
$$
**Estimating the weight norm**

Given the two property, if we assume $\langle u_t, W_t\rangle = 0$ and $\| u_t \|_F= U$, then we have the following equations following Pythagorean theorem (Gougu theorem勾股定理), 
$$
\begin{align*}
&W_{t + 1} = (1 - \eta \lambda) W_t - \eta u_t \\
\implies & \|W_{t + 1} \|_F^2 = (1 - \eta \lambda)^2 \| W_t \|_F^2 + \eta^2 \| u_t \|_F^2 \\
\implies & \|W_{t + 1} \|_F^2 = (1 - \eta \lambda)^2 \| W_t \|_F^2 + \eta^2 U^2
\end{align*}
$$
Without weight decay,  the weight norm will monotonously increase with respect to $t$
$$
\| W_{t} \|_{F} = \sqrt{t \eta^2 U^2 + \|W_0\|_F^2}.
$$
With weight decay, in contrast, the weight norm will converge to a constant
$$
\| W_{\infty} \|_F = \sqrt{\frac{\eta}{\lambda(2-\eta\lambda)}} U$$
 
$$ \| W_{t} \|_{F}  = \sqrt{ \| W_{\infty} \|_F^2 + (1 - \eta\lambda)^{2t} (\|W_0 \|_F^2 -  \| W_{\infty} \|_F^2) }
$$

[TBD: A figure of right triangular]

### Weight Decay Controls The Effective Learning Rate

Combining the two previous two paragraphs, we can show that the intrinsic learning rate schedule is actually a function of both learning rate and weight decay.


Without weight decay, we have showed that $\|W_t\|_F \approx \sqrt{t} \eta_t U$ and therefore the
$$
\eta_{eff,t} = \frac{\eta_t}{\|W_t\|_F} \|u_t\|_F \approx \sqrt{\frac{1}{t}}.
$$
With weight decay, we can show that $\|W_t\|_F \approx \sqrt{\frac{\eta_t}{2\lambda}} U$ and therefore
$$
\eta_{eff,t} = \frac{\eta_t}{\|W_t\|_F} \|u_t\|_F \approx \sqrt{2\eta_t \lambda}.
$$

[TBD: summarization]

## Explaining Empirical Phenomenon 

The classic saying goes "all models are wrong, but some are useful". In deep learning theory, whether a theory is useful should be judged by two perspectives:

1. Can it predict empirical observation?
2. Can it motivate algorithm that works better?

We will show evidence that for the theory we provide above, the answer is yes for both perspectives.

In this section, we will provide a list of empirical phenomenon that our model can explained.

**Phenomenon 1:** Weight norm track learning rate warmup and decay throughout training and gradient norm increases through training
**Phenomenon 4:**  Fixing the learning rate $\eta$, the loss decreases faster before learning rate decay to 0 when there is no weight decay but eventually result in a higher loss.
[Merged]



Comparing the effective learning rate, we know that the run without weight decay has fast decreasing learning rate compared to the run with weight decay. As the river valley paper show e.t.c


We observe this phenomenon for all sort of optimizers in our previous paper:

![image-20251025225246441](https://hackmd.io/_uploads/ByDtRP2All.png)

As we have derived 
$$
\|W_t\|_F \approx \sqrt{\eta/\lambda}
$$
The middle figure is just a consequence of it.

Further, for scale invariant loss
$$
L(cW + c\epsilon) - L(cW) = L(W + \epsilon) - L(W) \implies \nabla_W L(cW) = \nabla_W  L(W) / c.
$$

[TBD： Explain this means that the weight norm larger => gradient norm smaller]

Therefore, the gradient norm 
$$
\| G_t \|_F \propto \frac{1}{\|W_t\|_F} \propto \sqrt{\lambda/\eta}
$$
Naturally increase over time.

**Phenomenon 2:** When $\eta \lambda$ is fixed, the model trained with AdamW converge to essentially the same loss. At the same time, the weight norm of each weight matrices is proportional to learning rate.

![loss_vs_lr](https://hackmd.io/_uploads/H1rvX_hAlg.png)


[![image-20251025231214867](https://hackmd.io/_uploads/rydgy_n0le.png)
](https://)[](https://)






(The plot here needs to be improved but I am tired)

**Phenomenon 3:**  Contrary to the original MuP prediction, hyperparameter transfer is not sensitive to weight scale at initialization but it is sensitive to how to scale weight decay

(some recent paper)


## Using Hyperball Optimization to Remove Weight Decay 

Our derivation hints that weight decay is a lagged version of this update rule
$$
\hat W_{t + 1} \approx \mathrm{Norm}(\hat W_t - \eta_{eff,t} \mathrm{Norm}(u_t)).
$$



It is lagged because (i) it takes time for the weight norm to converge; and (ii) the update norm still vary between steps so the effective learning rate are actually affected

We propose to just do this precisely and fully eliminate weight decay
$$
 W_{t + 1} = \mathrm{Norm}(W_t - \eta_{eff,t} \mathrm{Norm}(u_t)).
$$

$$
 W_{t + 1} = \mathrm{Norm}(W_t - \eta_{eff,t} \mathrm{Norm}(u_t) \|W_0\|) \|W_0\| .
$$

[TBD: Explain why this is different from projected AdamW]

[TBD: Explain this is done per matrix]
We empirically observe two benefits of using this method

### Empirical Speedup On Marin Speedrun

![speerun](https://hackmd.io/_uploads/SJz-EO2Cee.png)


### Hyperparameter Transfer to 8x wider / 8x deeper model without pain

On Gemma-like architecture, we show that our method allow for almost precise learning rate transfer (within 1.4x) .

We sweep hyperparameter on a 4-layer 128-dimension 30M language models.



Width scaling

![loss_vs_lr_layers4_adamh](https://hackmd.io/_uploads/B1TVNH-y-x.png)
![loss_vs_lr_layers4_muonh](https://hackmd.io/_uploads/H16ENBZ1Zl.png)

Depth scaling

![loss_vs_lr_hidden128_adamh](https://hackmd.io/_uploads/HyaNNBb1-e.png)
![loss_vs_lr_hidden128_muonh](https://hackmd.io/_uploads/ByTVEH-k-x.png)


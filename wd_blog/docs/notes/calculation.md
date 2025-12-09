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

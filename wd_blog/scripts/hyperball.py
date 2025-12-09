import torch
from torch.optim import Optimizer
import math

class HyperballAdam(Optimizer):
    """
    Implements Hyperball Optimization with Adam as the base update mechanism.
    
    Theory:
        W_{t+1} = Norm(W_t - lr * Norm(u_t)) * R
        
    Where:
        u_t is the standard Adam update direction.
        R is the initial Frobenius norm of the weights (preserved throughout training).
        Norm(x) projects x to the unit sphere.
        
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): The effective step size on the sphere (angular change). 
                              Note: This behaves differently than standard Adam LR.
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
    """

    def __init__(self, params, lr=1e-3, lr_1d=None, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if lr_1d is None:
            lr_1d = lr  # Default to same learning rate as 2D+ parameters
        if not 0.0 <= lr_1d:
            raise ValueError("Invalid 1D learning rate: {}".format(lr_1d))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
            
        defaults = dict(lr=lr, lr_1d=lr_1d, betas=betas, eps=eps, weight_decay=weight_decay)
        super(HyperballAdam, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    
                    # [Hyperball Critical Step] Record the initial radius R (only for 2D+ params)
                    if p.dim() >= 2:
                        state['initial_norm'] = p.norm()

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                
                # --- 1. Compute Adam Update (u_t) ---
                
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                step_size_correction = math.sqrt(bias_correction2) / bias_correction1
                
                # --- 2. Apply Different Logic Based on Dimensionality ---
                
                if p.dim() < 2:
                    # Apply regular AdamW for 1D parameters (biases, layernorm parameters)
                    # AdamW update: p = p - lr * (m_t / (sqrt(v_t) + eps)) - lr * weight_decay * p
                    
                    # Use lr_1d for 1D parameters
                    lr_1d = group['lr_1d']
                    
                    # Weight decay
                    if group['weight_decay'] > 0:
                        p.mul_(1 - lr_1d * group['weight_decay'])
                    
                    # Adam update
                    adam_update = (exp_avg / denom) * step_size_correction
                    p.add_(adam_update, alpha=-lr_1d)
                    
                else:
                    # Apply Hyperball optimization for 2D+ parameters (weight matrices)
                    adam_update = (exp_avg / denom) * step_size_correction

                    # Calculate ||u_t||_F
                    update_norm = adam_update.norm()

                    # Normalize update direction: Norm(u_t)
                    # Avoid division by zero
                    normalized_update = adam_update / (update_norm + 1e-12)

                    # Apply update in the tangent space (approx): 
                    # W_temp = W_t - lr * Norm(u_t)
                    p.add_(normalized_update, alpha=-group['lr'])

                    # --- 3. Projection ---
                    
                    # W_{t+1} = Norm(W_temp) * R
                    # Project current weights back to the fixed radius R
                    p.mul_(state['initial_norm'] / (p.norm() + 1e-12))

        return loss


# write a simple test case for training 2 layer MLP with 128 dimension on parity classification task with normalization layer (such as RMSNorm)

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.rmsnorm1 = nn.RMSNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.rmsnorm1(x)
        x = F.relu(x)
        x = self.fc2(x) 
        return x


def generate_parity_data(n_samples, dim):
    """Generate binary vectors and their parity labels."""
    # Generate random binary vectors
    X = torch.randint(0, 2, (n_samples, dim)).float()
    # Compute parity: 1 if odd number of 1s, 0 if even
    y = (X.sum(dim=1) % 2).long()
    return X, y


if __name__ == "__main__":
    # Hyperparameters
    input_dim = 10
    hidden_dim = 128
    output_dim = 2
    n_train_samples = 1000
    n_test_samples = 200
    batch_size = 32
    n_epochs = 50
    lr = 0.01  # Learning rate for 2D+ parameters (weight matrices)
    lr_1d = 0.03 # Learning rate for 1D parameters (biases, layer norms)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Generate data
    X_train, y_train = generate_parity_data(n_train_samples, input_dim)
    X_test, y_test = generate_parity_data(n_test_samples, input_dim)
    
    # Create model and optimizer
    model = MLP(input_dim, hidden_dim, output_dim)
    optimizer = HyperballAdam(model.parameters(), lr=lr, lr_1d=lr_1d, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print(f"Training MLP with HyperballAdam on parity classification task")
    print(f"Input dim: {input_dim}, Hidden dim: {hidden_dim}")
    print(f"Training samples: {n_train_samples}, Test samples: {n_test_samples}")
    print(f"LR for 2D+ params: {lr}, LR for 1D params: {lr_1d}")
    print("-" * 60)
    
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Mini-batch training
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(batch_y).sum().item()
            total += batch_y.size(0)
        
        train_acc = 100. * correct / total
        avg_loss = total_loss / (len(X_train) // batch_size)
        
        # Evaluation on test set
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test)
                test_loss = criterion(test_outputs, y_test)
                _, test_predicted = test_outputs.max(1)
                test_acc = 100. * test_predicted.eq(y_test).sum().item() / len(y_test)
            
            print(f"Epoch {epoch+1:3d}/{n_epochs} | Train Loss: {avg_loss:.4f} | "
                  f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        _, test_predicted = test_outputs.max(1)
        test_acc = 100. * test_predicted.eq(y_test).sum().item() / len(y_test)
    
    print("-" * 60)
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    
    # Print parameter information
    print("\nParameter dimensions and learning rates:")
    for name, param in model.named_parameters():
        if param.dim() < 2:
            opt_type = f"AdamW (lr={lr_1d})"
        else:
            opt_type = f"Hyperball (lr={lr})"
        print(f"  {name:20s}: {str(param.shape):20s} -> {opt_type}")



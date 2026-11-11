# --- cell 1 ---
import torch
import torch.nn as nn

def generate_adversarial_example(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    criterion: nn.Module
) -> torch.Tensor:
    # Make a copy of input image
    x_adv = x.clone().detach()
    x_adv.requires_grad = True

    # Get model prediction
    output = model(x_adv)

    # Calculate loss
    loss = criterion(output, y)

    # Clear old gradients
    model.zero_grad()

    # Calculate gradients
    loss.backward()

    # Get gradient of input
    gradient = x_adv.grad.data

    # Create adversarial image
    x_adv = x_adv + epsilon * gradient.sign()

    # Keep pixel values between 0 and 1
    x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv.detach()
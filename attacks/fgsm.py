'''
    Fast Gradient Sign Method (FGSM) attack
'''

import torch
import torch.nn.functional as F

def fgsm(model, data, target, epsilon = 8/255, data_min=0, data_max=1):

    model.eval()
    # perturbed_data = copy.deepcopy(data)
    perturbed_data = data.clone()

    perturbed_data.requires_grad = True

    output = model(perturbed_data)
    loss = F.cross_entropy(output, target)

    if perturbed_data.grad is not None:
        perturbed_data.grad.data.zero_()

    model.zero_grad()

    loss.backward()

    # Collect the element-wise sign of the data gradient
    sign_data_grad = perturbed_data.grad.data.sign()
    perturbed_data.requires_grad = False

    with torch.no_grad():
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_data += epsilon*sign_data_grad
        # Adding clipping to maintain [min,max] range, default 0,1 for image
        perturbed_data.clamp_(data_min, data_max)

    return perturbed_data

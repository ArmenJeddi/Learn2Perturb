'''
   Learn2Perturb train function
'''

import torch
from attacks.pgd import pgd

def train(model, trainloader, epoch, optimizer1, optimizer2, criterion, layers, model_sigma_maps, args):
    model.train()

    if epoch < args.noise_add_delay:
        model[1].put_noise(False)
        for layer in layers:
            for block in layer:
                block.put_noise(False)
    else:
        model[1].put_noise(True)
        for layer in layers:
            for block in layer:
                block.put_noise(True)
        
        harmonic = 0.0
        for i in range(1, epoch - args.noise_add_delay +1):
            harmonic += 1 / i

    if epoch in args.lr_schedule:
        for g in optimizer1.param_groups:
            g['lr'] *= 0.1
        for g in optimizer2.param_groups:
            g['lr'] *= 0.1
    
    w_loss = 0.0
    theta_loss = 0.0
    total_loss = 0.0
    
    for itr, data in enumerate(trainloader):

        if epoch >= args.noise_add_delay:
            coef = args.gamma / harmonic
            sigma_map1 = model_sigma_maps[0]
            reg_term = -coef * torch.sqrt(sigma_map1)
            loss2 = reg_term.sum()
            for sigma_map in model_sigma_maps[1:]:
                reg_term = -coef * torch.sqrt(sigma_map)
                reg_loss = reg_term.sum()
                loss2 += reg_loss

            theta_loss = -loss2.item()
            
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

        inputs, labels = data[0], data[1]
        if torch.cuda.is_available():
            inputs.cuda()
            labels.cuda()

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        pred_target = outputs.max(1, keepdim=True)[1].squeeze(-1)

        total_loss += loss.item()

        if epoch >= args.adv_train_delay:
            perturbed_data = pgd(model, inputs, pred_target)
            output_adv = model(perturbed_data)
            loss_adv = criterion(output_adv, labels)
            loss = 0.5 * loss + 0.5 * loss_adv

        w_loss += loss.item()

        optimizer1.zero_grad()
        if epoch >= args.noise_add_delay:
            optimizer2.zero_grad()

        loss.backward()
        optimizer1.step()
        if epoch >= args.noise_add_delay:
            optimizer2.step()

        for sigma_map in model_sigma_maps:
            sigma_map.data = torch.clamp(sigma_map.data, 0.01)


        if itr % 100 == 99:    # print every 2000 mini-batches
            print('        [%d, %5d] total-loss: %.9f      w-loss: %.9f      theta-loss: %.9f' %
                    (epoch + 1, itr, total_loss / 100, w_loss / 100, theta_loss / 100))
            
            total_loss = 0.0
            w_loss = 0.0
            theta_loss = 0.0


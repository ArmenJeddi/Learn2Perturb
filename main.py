'''
    Reference implementation of Learn2Perturb. 
    Author: Ahmadreza Jeddi
    For more details, refer to the paper:
    Learn2Perturb: an End-to-end Feature Perturbation Learning to Improve Adversarial Robustness
    Ahmadreza Jeddi, Mohammad Javad Shafiee, Michelle Karg, Christian Scharfenberger, Alexander Wong
    Computer Vision and Pattern Recogniton (CVPR), 2020
'''

import argparse
import torchvision
import torchvision.transforms as transforms
import models
from models.normalizer import Normalize_layer
from train import train
from evaluate import evaluate

import os

def parse_args():
    '''
        parses the learn2perturb arguments
    '''
    parser = argparse.ArgumentParser(description="Learn2Perturb for adversarial robustness")

    # dataset and model config
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100'])
    parser.add_argument('--baseline', type=str, choices=['resnet_v1', 'resnet_v2'], default='resnet_v1')
    parser.add_argument('--res_v1_depth', type=int, default=20, help='depth of the res v1')
    parser.add_argument('--res_v2_num_blocks', type=int, nargs=4, default=[2,2,2,2], help='num blocks for each of the four layers of Res V2')

    # training optimization parameters
    parser.add_argument('--epochs', type=int, default=350, help='training epochs number')
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='The Learning Rate.')
    parser.add_argument('--lr_schedule', type=int, nargs='+', default=[150, 250], help='epochs in which lr is decreased')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--model_base', type=str, required=True, help='path to where save the models')

    # Learn2Perturb parameters
    parser.add_argument('--noise_add_delay', type=int, default=10, help='number of epochs to delay noise injection')
    parser.add_argument('--adv_train_delay', type=int, default=20, help='number of epochs to delay adversarial training')
    parser.add_argument('--gamma', type=float, default=1e-4, help='parameter gamma in equation (7)')

    return parser.parse_args()


def main(args):
    '''
        pipeline for training and evaluating robust deep convolutional models with Learn2Perturb
    '''

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    else: # cifar100
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)    
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    if args.baseline == 'resnet_v1':
        net = models.resnet_v1.l2p_resnet_v1(depth= args.res_v1_depth, num_classes= 10)
    else:
        net = models.resnet_v2.l2p_resnet_v2(num_blocks= args.res_v2_num_blocks, num_classes= 10)

    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    model = torch.nn.Sequential(
        Normalize_layer(mean,std),
        net
    )

    criterion = torch.nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()
    
    if args.baseline == 'resnet_v1':
        layers = [net.stage_1, net.stage_2, net.stage_3]
    else:
        layers = [net.layer1, net.layer2, net.layer3, net.layer4]

    model_sigma_maps = []
    model_sigma_maps.append(net.cn1_sigma_map)

    for layer in layers:
        for block in layer:
            model_sigma_maps.append(block.sigma_map)

    normal_param = [
        param for name, param in model.named_parameters()
        if not 'sigma' in name
    ]

    sigma_param = [
        param for name, param in model.named_parameters()
        if 'sigma' in name
    ]

    optimizer1 = torch.optim.SGD(normal_param, 
        lr=args.learning_rate,
        momentum=args.momentum, weight_decay=args.weight_decay,
        nesterov=True
    )

    optimizer2 = torch.optim.SGD(sigma_param, 
        lr=args.learning_rate,
        momentum=args.momentum, weight_decay=0,
        nesterov=True
    )

    ## create the folder to save models
    if not os.path.exists(args.model_base):
        os.makedirs(args.model_base)
    
    for epoch in range(args.epochs):
        print("epoch: {} / {} ...".format(epoch+1, args.epochs))
        print("    Training:")
        train(model, trainloader, epoch, optimizer1, optimizer2, criterion, layers, model_sigma_maps, args)
        print("    Evaluation:")
        evaluate(model, testloader, attack=None)
        evaluate(model, testloader, attack='pgd')
        evaluate(model, testloader, attack='fgsm')

        if (epoch +1) % 25 == 0:
            path = args.model_base + str(epoch + 1) + ".pt"
            torch.save(model, path)


if __name__ =='__main__':
    args = parse_args()
    main(args)


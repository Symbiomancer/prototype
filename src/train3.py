# coding=utf-8
from prototypical_batch_sampler import PrototypicalBatchSampler
from prototypical_loss import prototypical_loss as loss_fn
from omniglot_dataset import OmniglotDataset
from protonet import ProtoNet
from parser_util import get_parser

from tqdm import tqdm
import numpy as np
import torch
import os
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision

def init_seed(opt):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)


def init_dataset(opt, mode):
    dataset = OmniglotDataset(mode=mode, root=opt.dataset_root)
    n_classes = len(np.unique(dataset.y))
    if n_classes < opt.classes_per_it_tr or n_classes < opt.classes_per_it_val:
        raise(Exception('There are not enough classes in the dataset in order ' +
                        'to satisfy the chosen classes_per_it. Decrease the ' +
                        'classes_per_it_{tr/val} option and try again.'))
    return dataset


def init_sampler(opt, labels, mode):
    if 'train' in mode:
        classes_per_it = opt.classes_per_it_tr
        num_samples = opt.num_support_tr + opt.num_query_tr
    else:
        classes_per_it = opt.classes_per_it_val
        num_samples = opt.num_support_val + opt.num_query_val

    return PrototypicalBatchSampler(labels=labels,
                                    classes_per_it=classes_per_it,
                                    num_samples=num_samples,
                                    iterations=opt.iterations)


#def init_CIFAR_dataset(opt, mode):
 #   dataset = 


def init_dataloader(opt, mode):
    dataset = init_dataset(opt, mode)
    sampler = init_sampler(opt, dataset.y, mode)
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
    return dataloader


def init_protonet(opt):
    '''
    Initialize the ProtoNet
    '''
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    model = ProtoNet().to(device)
    print(model)
    return model


def init_optim(opt, model):
    '''
    Initialize optimizer
    '''
    return torch.optim.Adam(params=model.parameters(),
                            lr=opt.learning_rate)


def init_lr_scheduler(opt, optim):
    '''
    Initialize the learning rate scheduler
    '''
    return torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                           gamma=opt.lr_scheduler_gamma,
                                           step_size=opt.lr_scheduler_step)


def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)


def train(opt, tr_dataloader, model, optim, lr_scheduler, val_dataloader=None):
    '''
    Train the model with the prototypical learning algorithm
    '''

    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'

    if val_dataloader is None:
        best_state = None
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0

    best_model_path = os.path.join(opt.experiment_root, 'best_model.pth')
    last_model_path = os.path.join(opt.experiment_root, 'last_model.pth')

    for epoch in range(opt.epochs):
        print('=== Epoch: {} ==='.format(epoch))
        #tr_iter = iter(tr_dataloader)
        #tr_iter2 = iter(tr_dataloader)
        
        #print("after tr iter")
        model.train()
        #for i, batch in enumerate(tr_dataloader, 0):

        #print
        #n = 10
        #for i in range(n):
        #    print("inside first next iter")
        
         #   batch = next(tr_dataloader)
         #   print("batch outside loop: ", batch)

        for batch in tqdm(tr_dataloader, total=opt.iterations):

            #print("batch in main loop: ", batch)
            optim.zero_grad()
            x, y = batch
            x, y = x.to(device), y.to(device)
            #print("input before network: ", x)
            #print("input before network shape: ", x.shape)
            model_output = model(x)
            #print("model output: ", model_output)
            loss, acc = loss_fn(model_output, target=y,
                                n_support=opt.num_support_tr)
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
            train_acc.append(acc.item())
        avg_loss = np.mean(train_loss[-opt.iterations:])
        avg_acc = np.mean(train_acc[-opt.iterations:])
        print('Avg Train Loss: {}, Avg Train Acc: {}'.format(avg_loss, avg_acc))
        lr_scheduler.step()
        if val_dataloader is None:
            continue
        val_iter = iter(val_dataloader)
        model.eval()
        for batch in val_iter:
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            loss, acc = loss_fn(model_output, target=y,
                                n_support=opt.num_support_val)
            val_loss.append(loss.item())
            val_acc.append(acc.item())
        avg_loss = np.mean(val_loss[-opt.iterations:])
        avg_acc = np.mean(val_acc[-opt.iterations:])
        postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(
            best_acc)
        print('Avg Val Loss: {}, Avg Val Acc: {}{}'.format(
            avg_loss, avg_acc, postfix))
        if avg_acc >= best_acc:
            torch.save(model.state_dict(), best_model_path)
            best_acc = avg_acc
            best_state = model.state_dict()

    torch.save(model.state_dict(), last_model_path)


    for name in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
        save_list_to_file(os.path.join(opt.experiment_root,
                                       name + '.txt'), locals()[name])

    return best_state, best_acc, train_loss, train_acc, val_loss, val_acc


def test(opt, test_dataloader, model):
    '''
    Test the model trained with the prototypical learning algorithm
    '''
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    avg_acc = list()
    for epoch in range(10):
        test_iter = iter(test_dataloader)
        for batch in test_iter:
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            _, acc = loss_fn(model_output, target=y,
                             n_support=opt.num_support_val)
            avg_acc.append(acc.item())
    avg_acc = np.mean(avg_acc)
    print('Test Acc: {}'.format(avg_acc))

    return avg_acc


def eval(opt):
    '''
    Initialize everything and train
    '''
    options = get_parser().parse_args()

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)
    test_dataloader = init_dataset(options)[-1]
    model = init_protonet(options)
    model_path = os.path.join(opt.experiment_root, 'best_model.pth')
    model.load_state_dict(torch.load(model_path))

    test(opt=options,
         test_dataloader=test_dataloader,
         model=model)


def main():
    '''
    Initialize everything and train
    '''
    options = get_parser().parse_args()
    if not os.path.exists(options.experiment_root):
        os.makedirs(options.experiment_root)

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)

    #tr_dataloader = init_dataloader(options, 'train')
    #val_dataloader = init_dataloader(options, 'val')
    # trainval_dataloader = init_dataloader(options, 'trainval')
    #test_dataloader = init_dataloader(options, 'test')


    
    train_dataset = torchvision.datasets.MNIST(root='./files/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                 torchvision.transforms.ToTensor(),
                                 #torchvision.transforms.Normalize(
                                  # (0.1307,), (0.3081,))
                               ]))

    test_dataset = torchvision.datasets.MNIST(root='./files/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                 torchvision.transforms.ToTensor(),
                                 #torchvision.transforms.Normalize(
                                 #  (0.1307,), (0.3081,))
                                ]))

    """
    train_dataset = torchvision.datasets.CIFAR10(root='./files/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                 torchvision.transforms.ToTensor(),
                                 #=torchvision.transforms.Normalize(
                                  # (0.1307,), (0.3081,))
                               ]))

    test_dataset = torchvision.datasets.CIFAR10(root='./files/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                 torchvision.transforms.ToTensor(),
                                 #torchvision.transforms.Normalize(
                                 #  (0.1307,), (0.3081,))
                               ]))
    """
    #valid_dataset = torchvision.datasets.MNIST(root='./files/', train=True, download=True,
    #                           transform=torchvision.transforms.Compose([
     #                            torchvision.transforms.ToTensor(),
                                 #torchvision.transforms.Normalize(
                                  # (0.1307,), (0.3081,))
    #                           ]))

    random_seed=1337
    valid_perc = 0.1
    shuffle = True
    pin_memory=False
    num_workers=4

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_perc * num_train))
    batch_size=64 

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    #train_idx, valid_idx = indices[split:], indices[:split]
    #train_sampler = SubsetRandomSampler(train_idx)
    #valid_sampler = SubsetRandomSampler(valid_idx)

    


    """
    print("original train labels: ", train_dataset.targets)
    print("original train label size: ", train_dataset.targets.shape)

    

    train_size = int((1 - valid_perc) * len(train_dataset))
    val_size = len(train_dataset) - train_size


    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    print('train length: ', len(train_dataset))


    labels_train = train_dataset.targets
    labels_valid = valid_dataset.targets
    labels_test = test_dataset.targets

    Print("Labels train: ", labels_train)
    Print("Labels valid: ", labels_valid)
    Print("Labels test: ", labels_test)


    """

    labels_test = test_dataset.targets
    labels_train = train_dataset.targets

    train_sampler = init_sampler(options, labels_train, mode='train')
    #val_sampler = init_sampler(options, labels, mode='val')
    test_sampler = init_sampler(options, labels_test, mode='test')

    tr_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_sampler=train_sampler
    )

    #tr_dataloader = torch.utils.data.DataLoader(
     #train_dataset, batch_sampler= batch_size
    #)
    #val_dataloader = torch.utils.data.DataLoader(
    #    valid_dataset, batch_size=batch_size, sampler=val_sampler,
    #    num_workers=num_workers, pin_memory=pin_memory,
    #)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_sampler=test_sampler)

    val_dataloader = None

    print("Created CIFAR-10 train/test/val dataloaders.")
    print("Beginning training...")

    model = init_protonet(options)
    optim = init_optim(options, model)
    lr_scheduler = init_lr_scheduler(options, optim)

    print("Before calling train...")
    res = train(opt=options,
                tr_dataloader=tr_dataloader,
                val_dataloader=val_dataloader,
                model=model,
                optim=optim,
                lr_scheduler=lr_scheduler)
    best_state, best_acc, train_loss, train_acc, val_loss, val_acc = res
    print('Testing with last model..')
    test(opt=options,
         test_dataloader=test_dataloader,
         model=model)

    #model.load_state_dict(best_state)
    #print('Testing with best model..')
    #test(opt=options,
    #     test_dataloader=test_dataloader,
    #     model=model)

    # optim = init_optim(options, model)
    # lr_scheduler = init_lr_scheduler(options, optim)

    # print('Training on train+val set..')
    # train(opt=options,
    #       tr_dataloader=trainval_dataloader,
    #       val_dataloader=None,
    #       model=model,
    #       optim=optim,
    #       lr_scheduler=lr_scheduler)

    # print('Testing final model..')
    # test(opt=options,
    #      test_dataloader=test_dataloader,
    #      model=model)


if __name__ == '__main__':
    main()

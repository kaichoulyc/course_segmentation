import torch
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
from utils.losses import LossBinary, Loss, iou_pytorch
from utils.iou import compute_mask_iou
from utils.generator import claim_generator
import timeit
import datetime

from model.resnet import UNet_ResNet
from model.senet import UNet_SENet
from model.resnext import UNet_SeResnext

abs_path = '/home/kaichou/ssd/course'
train_path = os.path.join(abs_path, 'train')
valid_path = os.path.join(abs_path, 'valid')
test_path = os.path.join(abs_path, 'test')

epochs_df = pd.DataFrame(columns = ['Epoch', 'Train_loss', 'Val_loss', 'Val_IoU', 'Time'])

models = {
    'senet154': [UNet_SENet, 10, 'SENet154'],
    'resnext50': [UNet_SeResnext, 24, 'SEResNext50'],
    'resnext101': [UNet_SeResnext, 18, 'SEResNext101'],
    'resnet18': [UNet_ResNet, 34, 'ResNet18'],
    'resnet34': [UNet_ResNet, 26, 'ResNet34'],
    'resnet50': [UNet_ResNet, 22, 'ResNet50'],
    'resnet101': [UNet_ResNet, 20, 'ResNet101'],
    'resnet152': [UNet_ResNet, 18, 'ResNet152']
}

model_types = ['senet154', 'resnext50', 'resnext101', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

print('Chose one backbone')
for i, t in enumerate(model_types):
    print(str(i) + ': ' + t)
model_type = model_types[int(input())]

model_conf = models[model_type]

batch_size = model_conf[1]
num_workers = 10

uniq = datetime.datetime.now()
uniq_add = f'_{uniq.day}-{uniq.month}-{uniq.year}-{uniq.hour}-{uniq.minute}-{uniq.second}_'
way_to_info = 'information/' + model_type + '/' + model_conf[2] + uniq_add + 'inf.csv'

print('Enter learning rate:')
lr = float(input())
print('Enter amount of epochs:')
epochs = int(input())
path_for_weights = os.path.join('weights/', model_type)
path_for_model = os.path.join(path_for_weights, 'model')
path_for_optim = os.path.join(path_for_weights, 'optim')

train_loader = claim_generator(train_path, batch_size, num_workers, mode = 'train')
val_loader = claim_generator(valid_path, batch_size, num_workers, mode = 'test')

if model_type == 'senet154':
    model = model_conf[0]()
else:
    model = model_conf[0](encoder_type = model_type, pretrained=True)
    
criterions = {
    'lovasz': Loss(1, 2),
    'iou': compute_mask_iou
}

optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum=0.9, weight_decay=0.0005)

last_epoch = 0

print('Load weghts?')
weights = input()
if weights == 'Yes':
    print('Needed epoch:')
    last_epoch = int(input())
    model.load_state_dict(torch.load(os.path.join(path_for_model, model_conf[2] + f'_model{last_epoch}' + '.pth')))
    optimizer.load_state_dict(torch.load(os.path.join(path_for_optim, model_conf[2] + f'_optim{last_epoch}' + '.pth')))
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()


device = torch.device('cuda')
model = model.to(device)

def validating(model, criterions, val_loader, device):
    model.eval()
    with torch.no_grad():
        epoch_loss_b = 0
        schet = 0
        iou_full = 0
        for x, y in val_loader:
            schet += 1
            x = x.to(device)
            y = y.to(device)
            mask_pred = model(x)
            loss_b = criterions['lovasz'](mask_pred, y)
            epoch_loss_b += loss_b.item()
            iou_full += criterions['iou'](y.cpu().squeeze(1).numpy(), (torch.sigmoid(mask_pred) > 0.5).cpu().squeeze(1).numpy().astype(np.float))
    return epoch_loss_b / schet, iou_full / schet

def traininng(model, st_epoch, epochs, criterions, optimizer, train_loader,  path_for_model, path_for_optim, device, inf_df, model_conf, inf_way):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer, mode = 'min', verbose = True)
    for epoch in tqdm(range(st_epoch, epochs)):
        st = timeit.default_timer()
        model.train()
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        epoch_loss = 0
        schet = 0
        for x, y in train_loader:
            schet += 1
            x = x.to(device)
            with torch.no_grad():
                y = y.to(device)
            mask_pred = model(x)
            
            loss = criterions['lovasz'](mask_pred, y)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_loss, val_iou = validating(model, criterions, val_loader, device)
        scheduler.step(val_loss)
        print(f'Epoch finished ! Train Loss: {epoch_loss / schet}  Valid loss: {val_loss} Val IoUc: {val_iou}')
        
        
        if (epoch != 0) and (((epoch + 1) % 3) == 0):
            torch.save(model.state_dict(), os.path.join(path_for_model, model_conf[2] + f'_model{epoch + 1}.pth'))
            torch.save(optimizer.state_dict(), os.path.join(path_for_optim, model_conf[2] +  f'_optim{epoch + 1}.pth'))
        fin = timeit.default_timer() - st
        print(f'Time spent on epoch {fin}')
        inf_df.loc[epoch] = [epoch + 1, epoch_loss / schet, val_loss, val_iou, fin]
        inf_df.to_csv(inf_way)
        
traininng(model, last_epoch, epochs, criterions, optimizer, train_loader, path_for_model, path_for_optim, device, epochs_df, model_conf, way_to_info)
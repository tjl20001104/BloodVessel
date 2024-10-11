import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import cv2 as cv

def train(cfg, net, dataset, device):
    print('start training')
    trainer_net = optim.Adam(net.parameters(), lr=cfg['lr'],weight_decay=0.001)
    tubeloader = DataLoader(dataset, cfg['batch_size'], shuffle=True, 
                num_workers=cfg['num_data_workers'], drop_last=True)
    num_normal, num_exception = dataset.getlenofclass()
    weight_1 = num_exception/(num_normal+ num_exception)
    weight_2 = num_normal/(num_normal+ num_exception)
    weight = torch.tensor([weight_1,weight_2]).to(device)
    print('num_normal:' + str(num_normal))
    print('num_exception:' + str(num_exception))
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    for it in tqdm(range(0, cfg['epoch'] + 1), disable=None):
        log_loss = []

        for inputs, keypoints in tubeloader:
            inputs = inputs.to(device)
            keypoints = keypoints.to(device)
            key_pre = net(inputs).to(device)
            loss = criterion(key_pre,keypoints)

            trainer_net.zero_grad()
            loss.backward()
            trainer_net.step()

            log_loss.append(loss.detach().cpu().numpy())

        tqdm.write('loss = {}'.format(np.stack(log_loss).mean()))
        if it % cfg['save_freq'] == 0 and it > 0:
            model_path = cfg['model_path'].format(it + cfg['start_epoch'])
            torch.save(net.state_dict(),model_path)
    print('training done')



def test(cfg, net, dataset, device):
    #print('start testing')
    tubeloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=False)
    num_normal, num_exception = dataset.getlenofclass()
    #print(num_exception,num_normal)
    criterion = nn.CrossEntropyLoss()
    net.eval()
    log_loss = []
    idx = 0

    with torch.no_grad():
        for inputs, keypoints in tubeloader:
            inputs = inputs.to(device)
            keypoints = keypoints.to(device)
            key_pre = net(inputs)
            logsoftmax = nn.Softmax(dim=1)
            softmax_pre = logsoftmax(key_pre)
            loss = criterion(key_pre,keypoints)
            log_loss.append(loss.detach().cpu().numpy())
            inputs = inputs.detach().cpu().numpy()
            softmax_pre = softmax_pre.detach().cpu().numpy()
            softmax_pre = softmax_pre.astype(np.float32)
            real_v=keypoints.detach().cpu().numpy()
            real_v= real_v.astype(np.float32)
            pic_T3 = inputs[:,:3,:,:]
            pic_T5 = inputs[:,3:,:,:]
            pic_T3 = pic_T3.transpose(0,2,3,1)
            pic_T5 = pic_T5.transpose(0,2,3,1)
            '''
            for i in range(16):
                pic1 = pic_T3[i].copy()
                pic2 = pic_T5[i].copy()
                key = softmax_pre[i]
                cv.imwrite('result/{}_{}_{}_T3.png'.format(idx*16+i,'pre_'+str(key[1]),'real_'+str(real_v[i])), pic1)
            
            idx += 1
    '''
    print('loss = {}'.format(np.stack(log_loss).mean()))
    #print('testing done')
    
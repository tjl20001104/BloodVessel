import torch
from tubeDataSet import tubeDataset 
import mobilenetv3
from config import CONFIG
from train import train,test
from data_preprocess import preprocess 

device=  torch.device('cuda' if torch.cuda.is_available() else "cpu")
'''
preprocess(
    CONFIG['data_path'],
    CONFIG['box'],
    CONFIG['target_size']
)
'''
train_dataset = tubeDataset(
    CONFIG['data_path'],
    CONFIG['box'],
    CONFIG['target_size']
)


test_dataset = tubeDataset(
    CONFIG['data_path'],
    CONFIG['box'],
    CONFIG['target_size']
)

net = mobilenetv3.mobilenet_v3_small(
    num_classes= CONFIG['num_of_classes'],
    reduced_tail= 2
    ).to(device)

#防止训练异常中断需要重新从头训练
if CONFIG['start_epoch'] > 0:
    model_dict = torch.load(CONFIG['model_path'].format(CONFIG['start_epoch']))
    net.load_state_dict(model_dict, strict=False)
'''
test(CONFIG, net, train_dataset, device)
'''
for i in range(20,4001,20):
    model_dict = torch.load(CONFIG['model_path'].format(i))
    net.load_state_dict(model_dict, strict=False)
    test(CONFIG, net, train_dataset, device)




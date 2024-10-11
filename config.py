CONFIG = {
    'box' : (1000, 1600, 100, 1750),
    'target_size' : 256,
    'num_of_classes' : 2,

    #训练参数
    'start_epoch' : 0,
    'epoch' : 4000,
    'lr' : 1e-5,
    'batch_size': 16,
    'num_data_workers' : 0,

    #文件路径
    #normal_1/、exception_1/为训练集，normal_2/、exception_2/为测试集，在tubeDataSet.py文件中get_file_path函数修改   
    'data_path' : 'F:/6thdata/send/',
    'model_path' : 'models/model_{}.pt',
    'save_freq' : 20,
}
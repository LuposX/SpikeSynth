import os
import torch
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append(os.path.join(os.getcwd()))
sys.path.append('../')

class dataset(Dataset):
    def __init__(self, dataset, args, datapath, mode='train', temporal=False):
        self.args = args
        
        if datapath is None:
            datapath = os.path.join(args.DataPath, dataset)
        else:
            datapath = os.path.join(datapath, dataset)
        
        data = torch.load(datapath)
        
        X_train         = data['X_train']
        y_train         = data['y_train']
        X_valid         = data['X_valid']
        y_valid         = data['y_valid']
        X_test          = data['X_test']
        y_test          = data['y_test']

        if temporal:
            dimension = X_train.shape
            if len(dimension) == 3:
                pass
            elif len(dimension) == 2:
                self.N_time = args.N_time
                X_train = X_train.repeat(args.N_time, 1, 1).permute(1,2,0)
                X_valid = X_valid.repeat(args.N_time, 1, 1).permute(1,2,0)
                X_test  = X_test.repeat(args.N_time, 1, 1).permute(1,2,0)
        
        if mode == 'train':
            self.X_train    = torch.cat([X_train for _ in range(args.R_train)], dim=0).to(args.DEVICE)
            self.y_train    = torch.cat([y_train for _ in range(args.R_train)], dim=0).to(args.DEVICE)
        elif mode == 'valid':
            self.X_valid    = torch.cat([X_valid for _ in range(args.R_train)], dim=0).to(args.DEVICE)
            self.y_valid    = torch.cat([y_valid for _ in range(args.R_train)], dim=0).to(args.DEVICE)
        elif mode == 'test':
            self.X_test    = torch.cat([X_test for _ in range(args.R_test)], dim=0).to(args.DEVICE)
            self.y_test    = torch.cat([y_test for _ in range(args.R_test)], dim=0).to(args.DEVICE)
        
        self.data_name  = data['name']
        self.N_class    = data['n_class']
        self.N_feature  = data['n_feature']
        self.N_train    = X_train.shape[0]
        self.N_valid    = X_valid.shape[0]
        self.N_test     = X_test.shape[0]

        self.mode = mode
        
    @property
    def noisy_X_train(self):
        noise = torch.randn(self.X_train.shape) * self.args.InputNoise + 1.
        return self.X_train * noise.to(self.args.DEVICE) * 2.

    @property
    def noisy_X_valid(self):
        noise = torch.randn(self.X_valid.shape) * self.args.InputNoise +1.
        return self.X_valid * noise.to(self.args.DEVICE) * 2.
    
    @property
    def noisy_X_test(self):
        noise = torch.randn(self.X_test.shape) * self.args.IN_test + 1.
        return self.X_test * noise.to(self.args.DEVICE) * 2.
    
    
    def __getitem__(self, index):
        if self.mode == 'train':
            x = self.noisy_X_train[index,:]
            y = self.y_train[index]
        elif self.mode == 'valid':
            x = self.noisy_X_valid[index,:]
            y = self.y_valid[index]
        elif self.mode == 'test':
            x = self.noisy_X_test[index,:]
            y = self.y_test[index]
        return x, y
    
    def __len__(self):
        if self.mode == 'train':
            return self.N_train * self.args.R_train
        elif self.mode == 'valid':
            return self.N_valid * self.args.R_train
        elif self.mode == 'test':
            return self.N_test * self.args.R_test
        

def GetDataLoader(args, mode, path=None):
    normal_datasets = ['Dataset_acuteinflammation.ds',
                       'Dataset_balancescale.ds',
                       'Dataset_breastcancerwisc.ds',
                       'Dataset_cardiotocography3clases.ds',
                       'Dataset_energyy1.ds',
                       'Dataset_energyy2.ds',
                       'Dataset_iris.ds',
                       'Dataset_mammographic.ds',
                       'Dataset_pendigits.ds',
                       'Dataset_seeds.ds',
                       'Dataset_tictactoe.ds',
                       'Dataset_vertebralcolumn2clases.ds',
                       'Dataset_vertebralcolumn3clases.ds']

    temporized_datasets = normal_datasets

    split_manufacture = ['Dataset_acuteinflammation.ds',
                         'Dataset_acutenephritis.ds',
                         'Dataset_balancescale.ds',
                         'Dataset_blood.ds',
                         'Dataset_breastcancer.ds',
                         'Dataset_breastcancerwisc.ds',
                         'Dataset_breasttissue.ds',
                         'Dataset_ecoli.ds',
                         'Dataset_energyy1.ds',
                         'Dataset_energyy2.ds',
                         'Dataset_fertility.ds',
                         'Dataset_glass.ds',
                         'Dataset_habermansurvival.ds',
                         'Dataset_hayesroth.ds',
                         'Dataset_ilpdindianliver.ds',
                         'Dataset_iris.ds',
                         'Dataset_mammographic.ds',
                         'Dataset_monks1.ds',
                         'Dataset_monks2.ds',
                         'Dataset_monks3.ds',
                         'Dataset_pima.ds',
                         'Dataset_pittsburgbridgesMATERIAL.ds',
                         'Dataset_pittsburgbridgesSPAN.ds',
                         'Dataset_pittsburgbridgesTORD.ds',
                         'Dataset_pittsburgbridgesTYPE.ds',
                         'Dataset_seeds.ds',
                         'Dataset_teaching.ds',
                         'Dataset_tictactoe.ds',
                         'Dataset_vertebralcolumn2clases.ds',
                         'Dataset_vertebralcolumn3clases.ds']
    
    normal_datasets.sort()
    temporized_datasets.sort()
    split_manufacture.sort()
    
    if path is None:
        path = args.DataPath
    
    datasets = os.listdir(path)
    datasets = [f for f in datasets if (f.startswith('Dataset') and f.endswith('.ds'))]
    datasets.sort()

    
    if args.task=='normal':
        dataname = normal_datasets[args.DATASET]
        # data
        trainset  = dataset(dataname, args, path, mode='train')
        validset  = dataset(dataname, args, path, mode='valid')
        testset   = dataset(dataname, args, path, mode='test')

        # batch
        train_loader = DataLoader(trainset, batch_size=len(trainset))
        valid_loader = DataLoader(validset, batch_size=len(validset))
        test_loader  = DataLoader(testset,  batch_size=len(testset))
        
        # message
        info = {}
        info['dataname'] = trainset.data_name
        info['N_feature'] = trainset.N_feature
        info['N_class']   = trainset.N_class
        info['N_train']   = len(trainset)
        info['N_valid']   = len(validset)
        info['N_test']    = len(testset)
        
        if mode == 'train':
            return train_loader, info
        elif mode == 'valid':
            return valid_loader, info
        elif mode == 'test':
            return test_loader, info
    
    elif args.task=='split':
        train_loaders = []
        valid_loaders = []
        test_loaders  = []
        infos = []
        for dataname in split_manufacture:
            # data
            trainset  = dataset(dataname, args, path, mode='train')
            validset  = dataset(dataname, args, path, mode='valid')
            testset   = dataset(dataname, args, path, mode='test')
            # batch
            train_loaders.append(DataLoader(trainset, batch_size=len(trainset)))
            valid_loaders.append(DataLoader(validset, batch_size=len(validset)))
            test_loaders.append(DataLoader(testset,  batch_size=len(testset)))
            # message
            info = {}
            info['dataname'] = trainset.data_name
            info['N_feature'] = trainset.N_feature
            info['N_class']   = trainset.N_class
            info['N_train']   = len(trainset)
            info['N_valid']   = len(validset)
            info['N_test']    = len(testset)
            infos.append(info)

        if mode == 'train':
            return train_loaders, infos
        elif mode == 'valid':
            return valid_loaders, infos
        elif mode == 'test':
            return test_loaders, infos

    elif args.task=='temporized':
        dataname = temporized_datasets[args.DATASET]
        # data
        trainset  = dataset(dataname, args, path, mode='train', temporal=True)
        validset  = dataset(dataname, args, path, mode='valid', temporal=True)
        testset   = dataset(dataname, args, path, mode='test', temporal=True)

        # batch
        train_loader = DataLoader(trainset, batch_size=64)
        valid_loader = DataLoader(validset, batch_size=64)
        test_loader  = DataLoader(testset,  batch_size=64)
        
        # message
        info = {}
        info['dataname'] = trainset.data_name
        info['N_feature'] = trainset.N_feature
        info['N_class']   = trainset.N_class
        info['N_train']   = len(trainset)
        info['N_valid']   = len(validset)
        info['N_test']    = len(testset)
        info['N_time']    = trainset.N_time
        
        if mode == 'train':
            return train_loader, info
        elif mode == 'valid':
            return valid_loader, info
        elif mode == 'test':
            return test_loader, info
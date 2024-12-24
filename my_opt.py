import torch
import torch.nn as nn
import torch.nn.functional as F
from load_data import get_loader_flickr, get_loader_nus, get_loader_wiki
from torch.autograd import Variable
from models import *
from utils import compress, calculate_top_map, logger, p_topK, p_topK2
import numpy as np
import os.path as osp

class PCIRH:
    def __init__(self, log, config):
        self.config = config
        self.log = log

        if config.DATASET == 'MIRFlickr':
            dataloader, data_train = get_loader_flickr(config.BATCH_SIZE)
        elif config.DATASET == 'NUSWIDE':
            dataloader, data_train = get_loader_nus(config.BATCH_SIZE)
        elif config.DATASET == 'WIKI':
            dataloader, data_train = get_loader_wiki(config.BATCH_SIZE)

        self.I_tr, self.T_tr, self.L_tr = data_train

        self.train_num = self.I_tr.shape[0]
        self.img_feat_len = self.I_tr.shape[1]
        self.txt_feat_len = self.T_tr.shape[1]

        self.train_loader = dataloader['train']
        self.test_loader = dataloader['query']
        self.database_loader = dataloader['database']

        # 数据集的载入

        # img_norm = F.normalize(torch.Tensor(self.I_tr)).cuda()
        # txt_norm = F.normalize(torch.Tensor(self.T_tr)).cuda()
        # self.S_cmm = self.cal_similarity(img_norm, txt_norm)     # 调用了类自身的方法cal_similarity
        # 相似度矩阵的计算   训练实例5000 S_cmm 形状[5000,5000]

        
        self.Snet = S_net(code_len=self.config.HASH_BIT, ori_featI=self.img_feat_len, ori_featT=self.txt_feat_len).cuda()
        self.mynet = MyNet(code_len=self.config.HASH_BIT, ori_featI=self.img_feat_len, ori_featT=self.txt_feat_len, b1=self.config.b1, b2=self.config.b2).cuda()
        self.imgnet = Img_Net(code_len=self.config.HASH_BIT, img_feat_len=self.img_feat_len).cuda()
        self.txtnet = Txt_Net(code_len=self.config.HASH_BIT, txt_feat_len=self.txt_feat_len).cuda()
        # 网络结构的载入   import models里面的图文总网络


        self.opt_Snet = torch.optim.Adam(self.Snet.parameters(), lr=config.LR_SNet)   
        self.opt_mynet = torch.optim.Adam(self.mynet.parameters(), lr=config.LR_MyNet)
        #self.opt_imgnet = torch.optim.Adam(self.imgnet.parameters(), lr=config.LR_IMG)
        #self.opt_txtnet = torch.optim.Adam(self.txtnet.parameters(), lr=config.LR_TXT)
        # 优化器的载入

        self.record_Lsmodel = []
        self.record_Lshfunc = []

    def train_method(self, epoch):
        coll_B = list([])
        record_index = list([])
        Ls_method = 0

        self.Snet.train()

        self.mynet.train()

        for No, (img, txt, _, index_) in enumerate(self.train_loader):

            img = Variable(img.cuda().to(torch.float))  # [batchsize,4096]
            txt = Variable(torch.FloatTensor(txt.numpy()).cuda())   # [batchsize,1386]
            
            imgM, txtM = self.Snet(img, txt)     # print("imgM shape is ",imgM.shape)

            S_batch = self.cal_similarity(img, txt, imgM, txtM)  

            HI, HT, B, DeI_feat, DeT_feat = self.mynet(img, txt, S_batch)

            #  S_batch.shape [batchsize,batchsize]
            # Iind1, Iind2 = self.NN_select(S_batch)    # Iind 是一维列表 batchsize个
            # Tind1, Tind2 = self.NN_select(S_batch)        
            # imgNN1, imgNN2 = img[Iind1, :], img[Iind2, :]
            # txtNN1, txtNN2 = txt[Tind1, :], txt[Tind2, :]
            # imgN = 0.7 * img + 0.2 * imgNN2 + 0.1 * imgNN1
            # txtN = 0.7 * txt + 0.2 * txtNN2 + 0.1 * txtNN1
            # HI, HT, H, B, DeI_feat, DeT_feat = self.mynet(imgN, txtN, S_batch)

            # coll_B.extend(B.cpu().data.numpy())
            # record_index.extend(index_) 


            
            self.opt_mynet.zero_grad()
            self.opt_Snet.zero_grad()
            

            loss_CCR = self.config.lambda1 * (F.mse_loss(DeI_feat, img) + F.mse_loss(DeT_feat, txt))
            loss_ACR = self.loss_method(HI, HT, S_batch) 
            loss_DIS = F.mse_loss(HI, B) + F.mse_loss(HT, B)
           


            loss_mynet = loss_CCR +  loss_DIS
            #loss_my_S_net = loss_ACR
            loss_model = loss_mynet 
            # Ls_method = (Ls_method + loss_model).item()

            
            
          
            loss_model.backward()
            self.opt_mynet.step()
            self.opt_Snet.step()

            if (No + 1) % (self.train_num // self.config.BATCH_SIZE / self.config.EPOCH_INTERVAL) == 0:
                self.log.info('Epoch [%d/%d], Iter [%d/%d] Ls1=%.4f, Ls2=%.4f, Ls3=%.4f'
                                 % (epoch + 1, self.config.NUM_EPOCH, No + 1, self.train_num // self.config.BATCH_SIZE,
                                     loss_CCR, loss_ACR, loss_DIS))

        # coll_B = np.array(coll_B)

        # self.record_Lsmodel.append(Ls_method)

        return self.Snet, self.mynet

    

    def performance_eval(self):

        self.log.info('--------------------Evaluation: mAP@50-------------------')
        self.Snet.eval().cuda()
        self.mynet.eval().cuda()

        re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress(self.database_loader, self.test_loader, self.Snet, self.mynet)

        MAP_I2T = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=50)
        MAP_T2I = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=50)

        return MAP_I2T, MAP_T2I, re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L

    def eval2(self):

        self.log.info('--------------------Evaluation: mAP@50-------------------')
        self.imgnet.eval().cuda()
        self.txtnet.eval().cuda()

        re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress(self.database_loader, self.test_loader, self.imgnet, self.txtnet)

        MAP_I2T = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=50)
        MAP_T2I = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=50)

        return MAP_I2T, MAP_T2I

    def NN_select(self, S):

        m, n1 = S.sort()
        ind1 = n1[:, -4:-1][:,0]
        ind2 = n1[:, -4:-1][:,1]

        return ind1, ind2

    def cal_similarity(self, F_I, F_T, iM, tM):   # 改，由原始图文特征一个S，图文进入trans出来各个块再计算一个S
        
        # iM [codelen, batchsize, feat]
        # F_I [batchsize,4096]
        # F_T [batchsize,1386]
            
        batch_size = F_I.size(0)
        size = batch_size
        # top_size = self.config.K

        F_I = F.normalize(F_I)
        S_I = F_I.mm(F_I.t())
        F_T = F.normalize(F_T)
        S_T = F_T.mm(F_T.t())

        S1 = self.config.a1 * S_I + (1 - self.config.a1) * S_T

        # m, n1 = S1.sort()
        # S1[torch.arange(size).view(-1, 1).repeat(1, top_size).view(-1), n1[:, :top_size].contiguous().view(-1)] = 0.
        # S2 = 2.0 / (1 + torch.exp(-S1)) - 1 + torch.eye(S1.size(0)).cuda()
        # S2 = (S2 + S2.t())/2


        # S = self.config.a2 * S1 + (1 - self.config.a2) * S2
        S = self.config.a2 * S1 

        return S

    def loss_method(self, HI, HT, S_batch):

        HI_norm = F.normalize(HI)
        HT_norm = F.normalize(HT)
        # H_norm = F.normalize(H)

        HI_HI = HI_norm.mm(HI_norm.t())
        HT_HT = HT_norm.mm(HT_norm.t())
        HI_HT = HI_norm.mm(HT_norm.t())
        # H_H = H_norm.mm(H_norm.t())

        loss_ACR = self.config.lambda2 * (
                F.mse_loss(S_batch, HI_HI) + F.mse_loss(S_batch, HT_HT) ) + F.mse_loss(S_batch, HI_HT)
        # loss_DIS = F.mse_loss(H, B)

        return loss_ACR 


    def save_checkpoints(self):
        file_name = self.config.DATASET + '_' + str(self.config.HASH_BIT) + 'bits.pth'
        ckp_path = osp.join(self.config.MODEL_DIR, file_name)
        obj = {
            'ImgNet': self.imgnet.state_dict(),
            'TxtNet': self.txtnet.state_dict(),
        }
        torch.save(obj, ckp_path)
        self.log.info('**********Save the trained model successfully.**********')


    def load_checkpoints(self, file_name):

        ckp_path = osp.join(self.config.MODEL_DIR, file_name)
        try:
            obj = torch.load(ckp_path, map_location=lambda storage, loc: storage.cuda())
            self.log.info('**************** Load checkpoint %s ****************' % ckp_path)
        except IOError:
            self.log.error('********** Fail to load checkpoint %s!*********' % ckp_path)
            raise IOError

        self.imgnet.load_state_dict(obj['ImgNet'])
        self.txtnet.load_state_dict(obj['TxtNet'])



# def train_Hashfunc(self, coll_B, record_index, epoch, S_batch):

#         self.imgnet.train()
#         self.txtnet.train()

#         Ls_hfunc = 0

#         B = torch.from_numpy(coll_B).cuda()

#         img = torch.Tensor(self.I_tr[record_index,:]).cuda()
#         txt = torch.Tensor(self.T_tr[record_index,:]).cuda()

#         img_norm = F.normalize(img)
#         txt_norm = F.normalize(txt)
#         print("img_norm shape is   ",img_norm.shape)

#         num_cyc = img_norm.shape[0] / self.config.BATCH_SIZE
#         num_cyc = int(num_cyc+1) if num_cyc-int(num_cyc)>0 else int(num_cyc)

#         for kk in range(num_cyc):
#             if kk+1 < num_cyc:
#                 img_batch = img_norm[kk * self.config.BATCH_SIZE:(kk + 1) * self.config.BATCH_SIZE, :]
#                 txt_batch = txt_norm[kk * self.config.BATCH_SIZE:(kk + 1) * self.config.BATCH_SIZE, :]
#                 print("img_batch shape is  ",img_batch.shape)
#                 B_batch = B[kk * self.config.BATCH_SIZE:(kk + 1) * self.config.BATCH_SIZE, :]
#                 S_batch = self.S_cmm[kk * self.config.BATCH_SIZE:(kk + 1) * self.config.BATCH_SIZE,
#                                    kk * self.config.BATCH_SIZE:(kk + 1) * self.config.BATCH_SIZE]
#             else:
#                 img_batch = img_norm[kk * self.config.BATCH_SIZE:, :]
#                 txt_batch = txt_norm[kk * self.config.BATCH_SIZE:, :]
#                 print("img_batch shape is  ",img_batch.shape)
#                 B_batch = B[kk * self.config.BATCH_SIZE:, :]
#                 S_batch = self.S_cmm[kk * self.config.BATCH_SIZE:, kk * self.config.BATCH_SIZE:]

#             hfunc_BI = self.imgnet(img_batch)
#             hfunc_BT = self.txtnet(txt_batch)

#             self.opt_imgnet.zero_grad()
#             self.opt_txtnet.zero_grad()

#             loss_f1 = F.mse_loss(hfunc_BI, B_batch) + F.mse_loss(hfunc_BT, B_batch) + F.mse_loss(hfunc_BI, hfunc_BT)

#             S_BI_BT = F.normalize(hfunc_BI).mm(F.normalize(hfunc_BT).t())
#             S_BI_BI = F.normalize(hfunc_BI).mm(F.normalize(hfunc_BI).t())
#             S_BT_BT = F.normalize(hfunc_BT).mm(F.normalize(hfunc_BT).t())
#             loss_f2 = F.mse_loss(S_BI_BT, S_batch) + F.mse_loss(S_BI_BI, S_batch) + F.mse_loss(S_BT_BT, S_batch)

#             loss_hfunc = loss_f1 + self.config.beta * loss_f2
#             Ls_hfunc = (Ls_hfunc + loss_hfunc).item()

#             loss_hfunc.backward()

#             self.opt_imgnet.step()
#             self.opt_txtnet.step()

#         self.log.info('Epoch [%d/%d], Ls_hfunc: %.4f' % (epoch + 1, self.config.NUM_EPOCH, loss_hfunc.item()))

#         self.record_Lshfunc.append(Ls_hfunc)

#         return self.imgnet, self.txtnet


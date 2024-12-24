import torch
import math
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
try:
    from torch.nn import TransformerEncoder, TransformerEncoderLayer
except:
    raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        pos_encoder = PositionalEncoding(d_model)
    Refer:
        https://github.com/pytorch/examples/blob/master/word_language_model/model.py
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)



class MyNet(nn.Module):
    def __init__(self, code_len, ori_featI, ori_featT, b1, b2):
        super(MyNet, self).__init__()
        self.code_len = code_len
        self.b1 = b1
        self.b2 = b2

        ''' IRR_img '''
        self.encoderIMG = nn.Sequential(
            nn.Linear(ori_featI, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )                                      # 图 的 编码器

        self.gcnI1 = nn.Linear(512, 512)
        self.BNI1 = nn.BatchNorm1d(512)
        self.actI1 = nn.ReLU(inplace=True)     # 图 的 图卷积

        self.img_fc = nn.Linear(512, code_len)   # 图  的  哈希层

        self.decoderIMG = nn.Sequential(
            nn.Linear(code_len, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Linear(512, ori_featI),
            nn.BatchNorm1d(ori_featI)
            )                                  # 图 的 解码器



        '''IRR_txt'''
        self.encoderTXT = nn.Sequential(
            nn.Linear(ori_featT, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )                                         # 文 的 编码器

        self.gcnT1 = nn.Linear(512, 512)
        self.BNT1 = nn.BatchNorm1d(512)
        self.actT1 = nn.ReLU(inplace=True)         # 文 的 图卷积  

        self.txt_fc = nn.Linear(512, code_len)      # 文  的  哈希层

        self.decoderTXT = nn.Sequential(
            nn.Linear(code_len, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Linear(512, ori_featT),
            nn.BatchNorm1d(ori_featT)
            )                                        # 文 的 解码器


        self.HIBN = nn.BatchNorm1d(code_len)
        self.HTBN = nn.BatchNorm1d(code_len)

        '''CMA'''
        # self.gcnJ1 = nn.Linear(512, 512)
        # self.BNJ1 = nn.BatchNorm1d(512)
        # self.actJ1 = nn.ReLU(inplace=True)
        # self.HJ = nn.Linear(512, code_len)
        # self.HC = nn.Linear(3 * code_len, code_len)   
        # self.HJBN = nn.BatchNorm1d(code_len)
        # self.HBN = nn.BatchNorm1d(code_len)

    def forward(self, XI, XT, affinity_A):    # affinity_A是图结构，跟S有关

        self.batch_num = XI.size(0)

        ''' IRR_img '''
        VI = self.encoderIMG(XI)
        VI = F.normalize(VI, dim=1)     # 得到encoder之后的  Vi

        VgcnI = self.gcnI1(VI)
        VgcnI = affinity_A.mm(VgcnI)
        VgcnI = self.BNI1(VgcnI)
        VgcnI = self.actI1(VgcnI)         # Vi  图卷积后得到了  VGi

        HI = self.HIBN(self.img_fc(VgcnI))    # 哈希层
        DeI_feat = self.decoderIMG(torch.tanh(HI))   # 哈希码重建后得到原始特征


        ''' IRR_txt '''
        VT = self.encoderTXT(XT)
        VT = F.normalize(VT, dim=1)        # 得到encoder之后的  Vt

        VgcnT = self.gcnT1(VT)
        VgcnT = affinity_A.mm(VgcnT)
        VgcnT = self.BNT1(VgcnT)
        VgcnT = self.actT1(VgcnT)         # Vt  图卷积后得到了  VGt

        HT = self.HTBN(self.txt_fc( VgcnT))    # 哈希层
        DeT_feat = self.decoderTXT(torch.tanh(HT))   # 哈希码重建后得到原始特征


        B = torch.sign(torch.tanh(self.b1 * HI + self.b2 * HT))

        return HI, HT, B, DeI_feat, DeT_feat

        #'''CMA'''
        # VC = torch.cat((VI, VT), 0)
        # II = torch.eye(affinity_A.shape[0], affinity_A.shape[1]).cuda()
        # S_cma = torch.cat((torch.cat((affinity_A, II), 1),
        #                     torch.cat((II, affinity_A), 1)), 0)

        # VJ1 = self.gcnJ1(VC)
        # VJ1 = S_cma.mm(VJ1)
        # VJ1 = self.BNJ1(VJ1)
        # VJ1 = VJ1[:self.batch_num, :] + VJ1[self.batch_num:, :]
        # VJ = self.actJ1(VJ1)              #  得到了  CSA  模块的  VJ

        # HJ = self.HJ(VJ)
        # HJ = self.HJBN(HJ)

        
        

        # H = torch.tanh(self.HBN(self.HC(torch.cat((HI, HJ, HT), 1))))

        # B = torch.sign(H)

        
    
class S_net(nn.Module):   # 再改>>>>>>
    def __init__(self, code_len, ori_featI, ori_featT):
        super(S_net,self).__init__()
        self.code_len = code_len
        self.common_dim = 128
        self.dropout = 0.5
        self.nhead = 1
        self.num_layer = 2

        self.icn = nn.Linear(ori_featI, self.common_dim)
        self.tcn = nn.Linear(ori_featT, self.common_dim)

        self.imageConcept = nn.Linear(self.common_dim, self.common_dim * self.code_len)
        self.textConcept = nn.Linear(self.common_dim, self.common_dim * self.code_len)

        self.imagePosEncoder = PositionalEncoding(d_model=self.common_dim, dropout=self.dropout)
        self.textPosEncoder = PositionalEncoding(d_model=self.common_dim, dropout=self.dropout)

        imageEncoderLayer = TransformerEncoderLayer(d_model=self.common_dim,
                                                    nhead=self.nhead,
                                                    dim_feedforward=self.common_dim,
                                                    activation='gelu',
                                                    dropout=self.dropout)
        imageEncoderNorm = nn.LayerNorm(normalized_shape=self.common_dim)
        self.imageTransformerEncoder = TransformerEncoder(encoder_layer=imageEncoderLayer, num_layers=self.num_layer, norm=imageEncoderNorm)

        textEncoderLayer = TransformerEncoderLayer(d_model=self.common_dim,
                                                   nhead=self.nhead,
                                                   dim_feedforward=self.common_dim,
                                                   activation='gelu',
                                                   dropout=self.dropout)
        textEncoderNorm = nn.LayerNorm(normalized_shape=self.common_dim)
        self.textTransformerEncoder = TransformerEncoder(encoder_layer=textEncoderLayer, num_layers=self.num_layer, norm=textEncoderNorm)


    def forward(self, image, text):
        
        self.batch_size = image.size(0)

        imageH = self.icn(image)
        textH = self.tcn(text)

        imageC = self.imageConcept(imageH).reshape(imageH.size(0), self.code_len, self.common_dim).permute(1, 0, 2) # (nbit, bs, dim)
        textC = self.textConcept(textH).reshape(textH.size(0), self.code_len, self.common_dim).permute(1, 0, 2) # (nbit, bs, dim)
        # 解耦层
        imageSrc = self.imagePosEncoder(imageC)
        textSrc = self.textPosEncoder(textC)
        # transformer层
        imageMemory = self.imageTransformerEncoder(imageSrc)
        textMemory = self.textTransformerEncoder(textSrc)

        return imageMemory, textMemory

    

class Img_Net(nn.Module):
    def __init__(self, code_len, img_feat_len):
        super(Img_Net, self).__init__()

        self.fc1 = nn.Linear(img_feat_len, 4096)
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout(0.3)
        self.tohash = nn.Linear(4096, code_len)
        self.tanh = nn.Tanh()

        torch.nn.init.normal_(self.tohash.weight, mean=0.0, std=1)

    def forward(self, x):
        feat = self.relu(self.fc1(x))
        hid = self.tohash(self.dp(feat))
        HI = self.tanh(hid)

        return HI

class Txt_Net(nn.Module):
    def __init__(self, code_len, txt_feat_len):
        super(Txt_Net, self).__init__()

        self.fc1 = nn.Linear(txt_feat_len, txt_feat_len)
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout(0.3)
        self.tohash = nn.Linear(txt_feat_len, code_len)
        self.tanh = nn.Tanh()

        torch.nn.init.normal_(self.tohash.weight, mean=0.0, std=1)

    def forward(self, x):
        feat = self.relu(self.fc1(x))
        hid = self.tohash(self.dp(feat))
        HT = self.tanh(hid)

        return HT
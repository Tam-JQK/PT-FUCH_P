import torch
from torch import nn
from torch.nn import functional as F


class ImageNet(nn.Module):
    def __init__(self, y_dim, bit, norm=True, mid_num1=1024*8, mid_num2=1024*8, hiden_layer=3):
        """
        :param y_dim: dimension of tags
        :param bit: bit number of the final binary code
        """
        super(ImageNet, self).__init__()
        self.module_name = "img_model"

        mid_num1 = mid_num1 if hiden_layer > 1 else bit
        modules = [nn.Linear(y_dim, mid_num1)]
        if hiden_layer >= 2:
            modules += [nn.ReLU(inplace=True)]
            pre_num = mid_num1
            for i in range(hiden_layer - 2):
                if i == 0:
                    modules += [nn.Linear(mid_num1, mid_num2), nn.ReLU(inplace=True)]
                else:
                    modules += [nn.Linear(mid_num2, mid_num2), nn.ReLU(inplace=True)]
                pre_num = mid_num2
            modules += [nn.Linear(pre_num, bit)]
        self.fc = nn.Sequential(*modules)
        #self.apply(weights_init)
        self.norm = norm

    def forward(self, x):
        out1 = self.fc(x)
        out = torch.tanh(out1)
        if self.norm:
            norm_x = torch.norm(out, dim=1, keepdim=True)
            out = out / norm_x
        return out1,out
    
    def freeze_grad(self):
        for p in self.parameters():
            p.requires_grad = False


# class ImageNet(nn.Module):
#     def __init__(self, code_len):
#         super(ImageNet, self).__init__()
#         self.fc1 = nn.Linear(4096, 4096)
#         self.fc_encode = nn.Linear(4096, code_len)

#         self.alpha = 1.0
#         self.dropout = nn.Dropout(p=0.5)
#         self.relu = nn.ReLU(inplace=True)
#        # torch.nn.init.normal(self.fc_encode.weight, mean=0.0, std= 0.1)  

#     def forward(self, x):

#         x = x.view(x.size(0), -1)

#         feat1 = self.relu(self.fc1(x))
#         #feat1 = feat1 + self.relu(self.fc2(self.dropout(feat1)))
#         hid = self.fc_encode(self.dropout(feat1))
#         code = torch.tanh(hid)

#         return code

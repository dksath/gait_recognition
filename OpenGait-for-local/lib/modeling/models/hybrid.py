import torch
import copy
import torch.nn as nn

from ..base_model import BaseModel
from ..modules import SeparateFCs, BasicConv2d, SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateBNNecks
from torch.autograd import Variable


class GaitSet(BaseModel):
    """
        GaitSet: Regarding Gait as a Set for Cross-View Gait Recognition
        Arxiv:  https://arxiv.org/abs/1811.06186
        Github: https://github.com/AbnerHqC/GaitSet
    """

    def build_network(self, model_cfg):
        in_c = model_cfg['in_channels']
        self.set_block1 = nn.Sequential(BasicConv2d(in_c[0], in_c[1], 5, 1, 2),
                                        nn.LeakyReLU(inplace=True),
                                        BasicConv2d(in_c[1], in_c[1], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2))

        self.set_block2 = nn.Sequential(BasicConv2d(in_c[1], in_c[2], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        BasicConv2d(in_c[2], in_c[2], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2))

        self.set_block3 = nn.Sequential(BasicConv2d(in_c[2], in_c[3], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        BasicConv2d(in_c[3], in_c[3], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True))

        self.gl_block2 = copy.deepcopy(self.set_block2)
        self.gl_block3 = copy.deepcopy(self.set_block3)

        self.set_block1 = SetBlockWrapper(self.set_block1)
        self.set_block2 = SetBlockWrapper(self.set_block2)
        self.set_block3 = SetBlockWrapper(self.set_block3)

        self.set_pooling = PackSequenceWrapper(torch.max)

        self.Head = SeparateFCs(**model_cfg['SeparateFCs'])

        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])
        self.TP = PackSequenceWrapper(torch.max)

        self.Backbone = self.get_backbone(model_cfg['backbone_cfg'])
        self.Backbone = SetBlockWrapper(self.Backbone)

        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])

        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        sils = ipts[0]  # [n, s, h, w]
        n, s, h, w = sils.size()
        
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(2)

        del ipts
        
        outs = self.set_block1(sils)
        gl = self.set_pooling(outs, seqL, dim=1)[0]
        gl = self.gl_block2(gl)

        outs = self.set_block2(outs)
        gl = gl + self.set_pooling(outs, seqL, dim=1)[0]
        gl = self.gl_block3(gl)

        outs = self.set_block3(outs)
        outs = self.set_pooling(outs, seqL, dim=1)[0]
        gl = gl + outs


        #Spatial transformer
        sps = gl.reshape(n*s, 16, 16)
        iden = Variable(torch.eye(16)).unsqueeze(0).repeat(n*s, 1, 1)   # [n*s, 16, 16]
        if sps.is_cuda:
            iden = iden.cuda()
        sps_trans = sps + iden   # [n*s, 16, 16]

        
        outs = self.Backbone(sils)  # [n, s, c, h, w]
        outs_n, outs_s, outs_c, outs_h, outs_w = outs.size()

        zero_tensor = Variable(torch.zeros((outs_n, outs_s, outs_c, outs_h, outs_h-outs_w)))
        if outs.is_cuda:
            zero_tensor = zero_tensor.cuda()
        outs = torch.cat([outs, zero_tensor], -1)    # [n, s, c, h, h]  [n, s, c, 16, 16]
        outs = outs.reshape(outs_n * outs_s * outs_c, outs_h, outs_h)   # [n*s*c, 16, 16]

        sps = sps_trans.unsqueeze(1).repeat(1, outs_c, 1, 1).reshape(outs_n * outs_s * outs_c, 16, 16)

        outs_trans = torch.bmm(outs, sps)
        outs_trans = outs_trans.reshape(outs_n, outs_s, outs_c, outs_h, outs_h)

        # Temporal Pooling, TP
        outs_trans = self.TP(outs_trans, seqL, dim=1)[0]  # [n, c, h, w]
        # Horizontal Pooling Matching, HPM
        feat = self.HPP(outs_trans)  # [n, c, p]
        feat = feat.permute(2, 0, 1).contiguous()  # [p, n, c]
        embed_1 = self.FCs(feat)  # [p, n, c]

        embed_2, logits = self.BNNecks(embed_1)  # [p+1, n, c]

        embed_1 = embed_1.permute(1, 0, 2).contiguous()  # [n, p+1, c]
        logits = logits.permute(1, 0, 2).contiguous()  # [n, p+1, c]

        n, s, _, h, w = sils.size()
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': sils.view(n*s, 1, h, w)
            },
            'inference_feat': {
                'embeddings': embed_1
            }
        }
        return retval

import torch
import torch.nn as nn

class WCT(nn.Module):
    def __init__(self):
        super(WCT,self).__init__()

    def style_params(self,activation):
        #print("watch out. dimension [1] should be the channel dimension: " + str(activation.size()))
        feature = activation.view(activation.size(1), -1).double()
        feature_size = feature.size()
        mean = torch.mean(feature, 1)
        feature = feature - mean.unsqueeze(1).expand_as(feature)
        conv = torch.mm(feature, feature.t()).div(feature_size[1] - 1).cpu()
        print("svd")
        _, e, v = torch.svd(conv, some=False)
        print("svd done")
        k = feature_size[0]
        for i in range(feature_size[0]):
            if e[i] < 0.00001:
                k = i
                break
        e, v = e.cuda(), v.cuda()
        d = (e[0:k]).pow(0.5).cuda()
        target_feature = torch.mm(v[:, 0:k], torch.diag(d))
        target_feature = torch.mm(target_feature,(v[:, 0:k].t())).float()
        return {"mean" : mean.float(), "target_feature" : target_feature}


    def content_coloring(self,content_activation, style_params, alpha):
        #print("watch out. dimension [1] should be the channel dimension: " + str(content_activation.size()))
        content = content_activation.view(content_activation.size(1), -1).float()
        content_size = content.size()
        mean = torch.mean(content, 1)
        mean = mean.unsqueeze(1).expand_as(content)
        content = content - mean
        content_conv = (torch.mm(content, content.t()).div(content_size[1] - 1) + torch.eye(content_size[0])).cpu()
        _, content_e, content_v = torch.svd(content_conv, some=False)

        content_k = content_size[0]
        for i in range(content_size[0]):
            if content_e[i] < 0.00001:
                content_k = i
                break

        content_e, content_v = content_e.cuda(), content_v.cuda()

        content_d = (content_e[0:content_k]).pow(-0.5)
        step_1 = torch.mm(content_v[:, 0:content_k], torch.diag(content_d))
        step_2 = torch.mm(step_1, (content_v[:, 0:content_k].t()))
        white_content = torch.mm(step_2, content)

        target_feature = torch.mm(style_params["target_feature"],white_content).cuda()
        target_feature = target_feature + style_params["mean"].unsqueeze(1).expand_as(target_feature).cuda()

        target_feature = target_feature.view_as(content_activation)
        target_feature = alpha * target_feature + (1.0 - alpha) * content_activation
        return target_feature.half()

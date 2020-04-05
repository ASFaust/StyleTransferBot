import torch.nn as nn
import torch

from utils import load_t7

class _fe(nn.Module):
    instance = None
    conv1 = None
    reflecPad1 = None
    conv2 = None
    relu2 = None
    reflecPad3 = None
    conv3 = None
    relu3 = None
    maxPool = None
    reflecPad4 = None
    conv4 = None
    relu4 = None
    reflecPad5 = None
    conv5 = None
    relu5 = None
    maxPool2 = None
    reflecPad6 = None
    conv6 = None
    relu6 = None
    reflecPad7 = None
    conv7 = None
    relu7 = None
    reflecPad8 = None
    conv8 = None
    relu8 = None
    reflecPad9 = None
    conv9 = None
    relu9 = None
    maxPool3 = None
    reflecPad10 = None
    conv10 = None
    relu10 = None
    reflecPad11 = None
    conv11 = None
    relu11 = None
    reflecPad12 = None
    conv12 = None
    relu12 = None
    maxPool4 = None
    reflecPad13 = None
    conv13 = None
    relu13 = None
    reflecPad14 = None
    conv14 = None
    relu14 = None

    def __init__(self):
        torch.set_default_tensor_type('torch.cuda.HalfTensor')
        super(_fe, self).__init__()
        if not _fe.instance is None:
            return
        _fe.instance = self
        _fe._static_init()

    @staticmethod
    def _static_init():

        vgg = load_t7("models/vgg_normalised_conv5_1.t7")
        torch.set_default_tensor_type('torch.cuda.HalfTensor')
        # vgg
        # 224 x 224
        _fe.conv1 = nn.Conv2d(3,3,1,1,0)
        _fe.conv1.weight = torch.nn.Parameter(torch.from_numpy(vgg[0]["w"]).cuda())
        _fe.conv1.bias = torch.nn.Parameter(torch.from_numpy(vgg[0]["b"]).cuda())
        _fe.reflecPad1 = nn.ReflectionPad2d((1,1,1,1))
        # 226 x 226

        _fe.conv2 = nn.Conv2d(3,64,3,1,0)
        _fe.conv2.weight = torch.nn.Parameter(torch.from_numpy(vgg[1]["w"]).cuda())
        _fe.conv2.bias = torch.nn.Parameter(torch.from_numpy(vgg[1]["b"]).cuda())
        _fe.relu2 = nn.ReLU(inplace=True)
        # 224 x 224

        _fe.reflecPad3 = nn.ReflectionPad2d((1,1,1,1))
        _fe.conv3 = nn.Conv2d(64,64,3,1,0)
        _fe.conv3.weight = torch.nn.Parameter(torch.from_numpy(vgg[2]["w"]).cuda())
        _fe.conv3.bias = torch.nn.Parameter(torch.from_numpy(vgg[2]["b"]).cuda())
        _fe.relu3 = nn.ReLU(inplace=True)
        # 224 x 224

        _fe.maxPool = nn.MaxPool2d(kernel_size=2,stride=2,return_indices = True)
        # 112 x 112

        _fe.reflecPad4 = nn.ReflectionPad2d((1,1,1,1))
        _fe.conv4 = nn.Conv2d(64,128,3,1,0)
        _fe.conv4.weight = torch.nn.Parameter(torch.from_numpy(vgg[3]["w"]).cuda())
        _fe.conv4.bias = torch.nn.Parameter(torch.from_numpy(vgg[3]["b"]).cuda())
        _fe.relu4 = nn.ReLU(inplace=True)
        # 112 x 112

        _fe.reflecPad5 = nn.ReflectionPad2d((1,1,1,1))
        _fe.conv5 = nn.Conv2d(128,128,3,1,0)
        _fe.conv5.weight = torch.nn.Parameter(torch.from_numpy(vgg[4]["w"]).cuda())
        _fe.conv5.bias = torch.nn.Parameter(torch.from_numpy(vgg[4]["b"]).cuda())
        _fe.relu5 = nn.ReLU(inplace=True)
        # 112 x 112

        _fe.maxPool2 = nn.MaxPool2d(kernel_size=2,stride=2,return_indices = True)
        # 56 x 56

        _fe.reflecPad6 = nn.ReflectionPad2d((1,1,1,1))
        _fe.conv6 = nn.Conv2d(128,256,3,1,0)
        _fe.conv6.weight = torch.nn.Parameter(torch.from_numpy(vgg[5]["w"]).cuda())
        _fe.conv6.bias = torch.nn.Parameter(torch.from_numpy(vgg[5]["b"]).cuda())
        _fe.relu6 = nn.ReLU(inplace=True)
        # 56 x 56

        _fe.reflecPad7 = nn.ReflectionPad2d((1,1,1,1))
        _fe.conv7 = nn.Conv2d(256,256,3,1,0)
        _fe.conv7.weight = torch.nn.Parameter(torch.from_numpy(vgg[6]["w"]).cuda())
        _fe.conv7.bias = torch.nn.Parameter(torch.from_numpy(vgg[6]["b"]).cuda())
        _fe.relu7 = nn.ReLU(inplace=True)
        # 56 x 56

        _fe.reflecPad8 = nn.ReflectionPad2d((1,1,1,1))
        _fe.conv8 = nn.Conv2d(256,256,3,1,0)
        _fe.conv8.weight = torch.nn.Parameter(torch.from_numpy(vgg[7]["w"]).cuda())
        _fe.conv8.bias = torch.nn.Parameter(torch.from_numpy(vgg[7]["b"]).cuda())
        _fe.relu8 = nn.ReLU(inplace=True)
        # 56 x 56

        _fe.reflecPad9 = nn.ReflectionPad2d((1,1,1,1))
        _fe.conv9 = nn.Conv2d(256,256,3,1,0)
        _fe.conv9.weight = torch.nn.Parameter(torch.from_numpy(vgg[8]["w"]).cuda())
        _fe.conv9.bias = torch.nn.Parameter(torch.from_numpy(vgg[8]["b"]).cuda())
        _fe.relu9 = nn.ReLU(inplace=True)
        # 56 x 56

        _fe.maxPool3 = nn.MaxPool2d(kernel_size=2,stride=2,return_indices = True)
        # 28 x 28

        _fe.reflecPad10 = nn.ReflectionPad2d((1,1,1,1))
        _fe.conv10 = nn.Conv2d(256,512,3,1,0)
        _fe.conv10.weight = torch.nn.Parameter(torch.from_numpy(vgg[9]["w"]).cuda())
        _fe.conv10.bias = torch.nn.Parameter(torch.from_numpy(vgg[9]["b"]).cuda())
        _fe.relu10 = nn.ReLU(inplace=True)
        # 28 x 28

        _fe.reflecPad11 = nn.ReflectionPad2d((1,1,1,1))
        _fe.conv11 = nn.Conv2d(512,512,3,1,0)
        _fe.conv11.weight = torch.nn.Parameter(torch.from_numpy(vgg[10]["w"]).cuda())
        _fe.conv11.bias = torch.nn.Parameter(torch.from_numpy(vgg[10]["b"]).cuda())
        _fe.relu11 = nn.ReLU(inplace=True)
        # 28 x 28

        _fe.reflecPad12 = nn.ReflectionPad2d((1,1,1,1))
        _fe.conv12 = nn.Conv2d(512,512,3,1,0)
        _fe.conv12.weight = torch.nn.Parameter(torch.from_numpy(vgg[11]["w"]).cuda())
        _fe.conv12.bias = torch.nn.Parameter(torch.from_numpy(vgg[11]["b"]).cuda())
        _fe.relu12 = nn.ReLU(inplace=True)
        # 28 x 28

        _fe.reflecPad13 = nn.ReflectionPad2d((1,1,1,1))
        _fe.conv13 = nn.Conv2d(512,512,3,1,0)
        _fe.conv13.weight = torch.nn.Parameter(torch.from_numpy(vgg[12]["w"]).cuda())
        _fe.conv13.bias = torch.nn.Parameter(torch.from_numpy(vgg[12]["b"]).cuda())
        _fe.relu13 = nn.ReLU(inplace=True)
        # 28 x 28

        _fe.maxPool4 = nn.MaxPool2d(kernel_size=2,stride=2,return_indices = True)
        # 14 x 14

        _fe.reflecPad14 = nn.ReflectionPad2d((1,1,1,1))
        _fe.conv14 = nn.Conv2d(512,512,3,1,0)
        _fe.conv14.weight = torch.nn.Parameter(torch.from_numpy(vgg[13]["w"]).cuda())
        _fe.conv14.bias = torch.nn.Parameter(torch.from_numpy(vgg[13]["b"]).cuda())
        _fe.relu14 = nn.ReLU(inplace=True)
        # 14 x 14

class level_1_encoder(_fe):
    def __init__(self):
        torch.set_default_tensor_type('torch.cuda.HalfTensor')
        super(level_1_encoder, self).__init__()

    def forward(self,x):
        torch.set_default_tensor_type('torch.cuda.HalfTensor')
        with torch.no_grad():
            out = style_features.conv1(x)
            out = style_features.reflecPad1(out)
            out = style_features.conv2(out)
            out_0 = style_features.relu2(out)
        return out_0


class level_2_encoder(_fe):
    def __init__(self):
        torch.set_default_tensor_type('torch.cuda.HalfTensor')
        super(level_2_encoder, self).__init__()

    def forward(self,x):
        torch.set_default_tensor_type('torch.cuda.HalfTensor')
        with torch.no_grad():
            out = style_features.conv1(x)
            out = style_features.reflecPad1(out)
            out = style_features.conv2(out)
            out_0 = style_features.relu2(out)
            out = style_features.reflecPad3(out_0)
            out = style_features.conv3(out)
            out = style_features.relu3(out)
            out,pool_idx = style_features.maxPool(out)
            out = style_features.reflecPad4(out)
            out = style_features.conv4(out)
            out_1 = style_features.relu4(out)
        return out_1

class level_3_encoder(_fe):
    def __init__(self):
        torch.set_default_tensor_type('torch.cuda.HalfTensor')
        super(level_3_encoder, self).__init__()

    def forward(self,x):
        torch.set_default_tensor_type('torch.cuda.HalfTensor')
        with torch.no_grad():
            out = style_features.conv1(x)
            out = style_features.reflecPad1(out)
            out = style_features.conv2(out)
            out_0 = style_features.relu2(out)
            out = style_features.reflecPad3(out_0)
            out = style_features.conv3(out)
            out = style_features.relu3(out)
            out,pool_idx = style_features.maxPool(out)
            out = style_features.reflecPad4(out)
            out = style_features.conv4(out)
            out_1 = style_features.relu4(out)
            out = style_features.reflecPad5(out_1)
            out = style_features.conv5(out)
            out = style_features.relu5(out)
            out,pool_idx2 = style_features.maxPool2(out)
            out = style_features.reflecPad6(out)
            out = style_features.conv6(out)
            out_2 = style_features.relu6(out)
        return out_2

class level_4_encoder(_fe):
    def __init__(self):
        torch.set_default_tensor_type('torch.cuda.HalfTensor')
        super(level_4_encoder, self).__init__()

    def forward(self,x):
        torch.set_default_tensor_type('torch.cuda.HalfTensor')
        with torch.no_grad():
            out = style_features.conv1(x)
            out = style_features.reflecPad1(out)
            out = style_features.conv2(out)
            out_0 = style_features.relu2(out)
            out = style_features.reflecPad3(out_0)
            out = style_features.conv3(out)
            out = style_features.relu3(out)
            out,pool_idx = style_features.maxPool(out)
            out = style_features.reflecPad4(out)
            out = style_features.conv4(out)
            out_1 = style_features.relu4(out)
            out = style_features.reflecPad5(out_1)
            out = style_features.conv5(out)
            out = style_features.relu5(out)
            out,pool_idx2 = style_features.maxPool2(out)
            out = style_features.reflecPad6(out)
            out = style_features.conv6(out)
            out_2 = style_features.relu6(out)
            out = style_features.reflecPad7(out_2)
            out = style_features.conv7(out)
            out = style_features.relu7(out)
            out = style_features.reflecPad8(out)
            out = style_features.conv8(out)
            out = style_features.relu8(out)
            out = style_features.reflecPad9(out)
            out = style_features.conv9(out)
            out = style_features.relu9(out)
            out,pool_idx3 = style_features.maxPool3(out)
            out = style_features.reflecPad10(out)
            out = style_features.conv10(out)
            out_3 = style_features.relu10(out)
        return out_3

class level_5_encoder(_fe):
    def __init__(self):
        torch.set_default_tensor_type('torch.cuda.HalfTensor')
        super(level_5_encoder, self).__init__()

    def forward(self,x):
        torch.set_default_tensor_type('torch.cuda.HalfTensor')
        with torch.no_grad():
            out = style_features.conv1(x)
            out = style_features.reflecPad1(out)
            out = style_features.conv2(out)
            out_0 = style_features.relu2(out)
            out = style_features.reflecPad3(out_0)
            out = style_features.conv3(out)
            out = style_features.relu3(out)
            out,pool_idx = style_features.maxPool(out)
            out = style_features.reflecPad4(out)
            out = style_features.conv4(out)
            out_1 = style_features.relu4(out)
            out = style_features.reflecPad5(out_1)
            out = style_features.conv5(out)
            out = style_features.relu5(out)
            out,pool_idx2 = style_features.maxPool2(out)
            out = style_features.reflecPad6(out)
            out = style_features.conv6(out)
            out_2 = style_features.relu6(out)
            out = style_features.reflecPad7(out_2)
            out = style_features.conv7(out)
            out = style_features.relu7(out)
            out = style_features.reflecPad8(out)
            out = style_features.conv8(out)
            out = style_features.relu8(out)
            out = style_features.reflecPad9(out)
            out = style_features.conv9(out)
            out = style_features.relu9(out)
            out,pool_idx3 = style_features.maxPool3(out)
            out = style_features.reflecPad10(out)
            out = style_features.conv10(out)
            out_3 = style_features.relu10(out)
            out = style_features.reflecPad11(out_3)
            out = style_features.conv11(out)
            out = style_features.relu11(out)
            out = style_features.reflecPad12(out)
            out = style_features.conv12(out)
            out = style_features.relu12(out)
            out = style_features.reflecPad13(out)
            out = style_features.conv13(out)
            out = style_features.relu13(out)
            out,pool_idx4 = style_features.maxPool4(out)
            out = style_features.reflecPad14(out)
            out = style_features.conv14(out)
            out_4 = style_features.relu14(out)
        return out_4

class style_features(_fe):
    def __init__(self):
        torch.set_default_tensor_type('torch.cuda.HalfTensor')
        super(style_features, self).__init__()

    def forward(self,x):
        torch.set_default_tensor_type('torch.cuda.HalfTensor')
        with torch.no_grad():
            out = style_features.conv1(x)
            out = style_features.reflecPad1(out)
            out = style_features.conv2(out)
            out_0 = style_features.relu2(out)
            out = style_features.reflecPad3(out_0)
            out = style_features.conv3(out)
            out = style_features.relu3(out)
            out,pool_idx = style_features.maxPool(out)
            out = style_features.reflecPad4(out)
            out = style_features.conv4(out)
            out_1 = style_features.relu4(out)
            out = style_features.reflecPad5(out_1)
            out = style_features.conv5(out)
            out = style_features.relu5(out)
            out,pool_idx2 = style_features.maxPool2(out)
            out = style_features.reflecPad6(out)
            out = style_features.conv6(out)
            out_2 = style_features.relu6(out)
            out = style_features.reflecPad7(out_2)
            out = style_features.conv7(out)
            out = style_features.relu7(out)
            out = style_features.reflecPad8(out)
            out = style_features.conv8(out)
            out = style_features.relu8(out)
            out = style_features.reflecPad9(out)
            out = style_features.conv9(out)
            out = style_features.relu9(out)
            out,pool_idx3 = style_features.maxPool3(out)
            out = style_features.reflecPad10(out)
            out = style_features.conv10(out)
            out_3 = style_features.relu10(out)
            out = style_features.reflecPad11(out_3)
            out = style_features.conv11(out)
            out = style_features.relu11(out)
            out = style_features.reflecPad12(out)
            out = style_features.conv12(out)
            out = style_features.relu12(out)
            out = style_features.reflecPad13(out)
            out = style_features.conv13(out)
            out = style_features.relu13(out)
            out,pool_idx4 = style_features.maxPool4(out)
            out = style_features.reflecPad14(out)
            out = style_features.conv14(out)
            out_4 = style_features.relu14(out)
        return out_0,out_1,out_2,out_3,out_4

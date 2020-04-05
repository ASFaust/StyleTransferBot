import torch.nn as nn
import torch

from utils import load_t7

class level_1_decoder(nn.Module):
    def __init__(self):
        super(level_1_decoder,self).__init__()
        d1 = load_t7("models/feature_invertor_conv1_1.t7")
        self.reflecPad2 = nn.ReflectionPad2d((1,1,1,1))
        # 226 x 226
        self.conv3 = nn.Conv2d(64,3,3,1,0)
        self.conv3.weight = torch.nn.Parameter(torch.from_numpy(d1[0]["w"]).cuda())
        self.conv3.bias = torch.nn.Parameter(torch.from_numpy(d1[0]["b"]).cuda())
        # 224 x 224

    def forward(self,x):
        out = self.reflecPad2(x)
        out = self.conv3(out)
        return out

class level_2_decoder(nn.Module):
    def __init__(self):
        super(level_2_decoder,self).__init__()
        # decoder
        d = load_t7("models/feature_invertor_conv2_1.t7")
        self.reflecPad5 = nn.ReflectionPad2d((1,1,1,1))
        self.conv5 = nn.Conv2d(128,64,3,1,0)
        self.conv5.weight = torch.nn.Parameter(torch.from_numpy(d[0]["w"]).cuda())
        self.conv5.bias = torch.nn.Parameter(torch.from_numpy(d[0]["b"]).cuda())
        self.relu5 = nn.ReLU(inplace=True)
        # 112 x 112

        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        # 224 x 224

        self.reflecPad6 = nn.ReflectionPad2d((1,1,1,1))
        self.conv6 = nn.Conv2d(64,64,3,1,0)
        self.conv6.weight = torch.nn.Parameter(torch.from_numpy(d[1]["w"]).cuda())
        self.conv6.bias = torch.nn.Parameter(torch.from_numpy(d[1]["b"]).cuda())
        self.relu6 = nn.ReLU(inplace=True)
        # 224 x 224

        self.reflecPad7 = nn.ReflectionPad2d((1,1,1,1))
        self.conv7 = nn.Conv2d(64,3,3,1,0)
        self.conv7.weight = torch.nn.Parameter(torch.from_numpy(d[2]["w"]).cuda())
        self.conv7.bias = torch.nn.Parameter(torch.from_numpy(d[2]["b"]).cuda())

    def forward(self,x):
        out = self.reflecPad5(x)
        out = self.conv5(out)
        out = self.relu5(out)
        out = self.unpool(out)
        out = self.reflecPad6(out)
        out = self.conv6(out)
        out = self.relu6(out)
        out = self.reflecPad7(out)
        out = self.conv7(out)
        return out

class level_3_decoder(nn.Module):
    def __init__(self):
        super(level_3_decoder,self).__init__()
        # decoder
        d = load_t7("models/feature_invertor_conv3_1.t7")
        self.reflecPad7 = nn.ReflectionPad2d((1,1,1,1))
        self.conv7 = nn.Conv2d(256,128,3,1,0)
        self.conv7.weight = torch.nn.Parameter(torch.from_numpy(d[0]["w"]).cuda())
        self.conv7.bias = torch.nn.Parameter(torch.from_numpy(d[0]["b"]).cuda())
        self.relu7 = nn.ReLU(inplace=True)
        # 56 x 56

        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        # 112 x 112

        self.reflecPad8 = nn.ReflectionPad2d((1,1,1,1))
        self.conv8 = nn.Conv2d(128,128,3,1,0)
        self.conv8.weight = torch.nn.Parameter(torch.from_numpy(d[1]["w"]).cuda())
        self.conv8.bias = torch.nn.Parameter(torch.from_numpy(d[1]["b"]).cuda())
        self.relu8 = nn.ReLU(inplace=True)
        # 112 x 112

        self.reflecPad9 = nn.ReflectionPad2d((1,1,1,1))
        self.conv9 = nn.Conv2d(128,64,3,1,0)
        self.conv9.weight = torch.nn.Parameter(torch.from_numpy(d[2]["w"]).cuda())
        self.conv9.bias = torch.nn.Parameter(torch.from_numpy(d[2]["b"]).cuda())
        self.relu9 = nn.ReLU(inplace=True)

        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
        # 224 x 224

        self.reflecPad10 = nn.ReflectionPad2d((1,1,1,1))
        self.conv10 = nn.Conv2d(64,64,3,1,0)
        self.conv10.weight = torch.nn.Parameter(torch.from_numpy(d[3]["w"]).cuda())
        self.conv10.bias = torch.nn.Parameter(torch.from_numpy(d[3]["b"]).cuda())
        self.relu10 = nn.ReLU(inplace=True)

        self.reflecPad11 = nn.ReflectionPad2d((1,1,1,1))
        self.conv11 = nn.Conv2d(64,3,3,1,0)
        self.conv11.weight = torch.nn.Parameter(torch.from_numpy(d[4]["w"]).cuda())
        self.conv11.bias = torch.nn.Parameter(torch.from_numpy(d[4]["b"]).cuda())

    def forward(self,x):
        out = self.reflecPad7(x)
        out = self.conv7(out)
        out = self.relu7(out)
        out = self.unpool(out)
        out = self.reflecPad8(out)
        out = self.conv8(out)
        out = self.relu8(out)
        out = self.reflecPad9(out)
        out = self.conv9(out)
        out = self.relu9(out)
        out = self.unpool2(out)
        out = self.reflecPad10(out)
        out = self.conv10(out)
        out = self.relu10(out)
        out = self.reflecPad11(out)
        out = self.conv11(out)
        return out

class level_4_decoder(nn.Module):
    def __init__(self):
        super(level_4_decoder,self).__init__()
        d = load_t7("models/feature_invertor_conv4_1.t7")
        # decoder
        self.reflecPad11 = nn.ReflectionPad2d((1,1,1,1))
        self.conv11 = nn.Conv2d(512,256,3,1,0)
        self.conv11.weight = torch.nn.Parameter(torch.from_numpy(d[0]["w"]).cuda())
        self.conv11.bias = torch.nn.Parameter(torch.from_numpy(d[0]["b"]).cuda())
        self.relu11 = nn.ReLU(inplace=True)
        # 28 x 28

        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        # 56 x 56

        self.reflecPad12 = nn.ReflectionPad2d((1,1,1,1))
        self.conv12 = nn.Conv2d(256,256,3,1,0)
        self.conv12.weight = torch.nn.Parameter(torch.from_numpy(d[1]["w"]).cuda())
        self.conv12.bias = torch.nn.Parameter(torch.from_numpy(d[1]["b"]).cuda())
        self.relu12 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad13 = nn.ReflectionPad2d((1,1,1,1))
        self.conv13 = nn.Conv2d(256,256,3,1,0)
        self.conv13.weight = torch.nn.Parameter(torch.from_numpy(d[2]["w"]).cuda())
        self.conv13.bias = torch.nn.Parameter(torch.from_numpy(d[2]["b"]).cuda())
        self.relu13 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad14 = nn.ReflectionPad2d((1,1,1,1))
        self.conv14 = nn.Conv2d(256,256,3,1,0)
        self.conv14.weight = torch.nn.Parameter(torch.from_numpy(d[3]["w"]).cuda())
        self.conv14.bias = torch.nn.Parameter(torch.from_numpy(d[3]["b"]).cuda())
        self.relu14 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad15 = nn.ReflectionPad2d((1,1,1,1))
        self.conv15 = nn.Conv2d(256,128,3,1,0)
        self.conv15.weight = torch.nn.Parameter(torch.from_numpy(d[4]["w"]).cuda())
        self.conv15.bias = torch.nn.Parameter(torch.from_numpy(d[4]["b"]).cuda())
        self.relu15 = nn.ReLU(inplace=True)
        # 56 x 56

        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
        # 112 x 112

        self.reflecPad16 = nn.ReflectionPad2d((1,1,1,1))
        self.conv16 = nn.Conv2d(128,128,3,1,0)
        self.conv16.weight = torch.nn.Parameter(torch.from_numpy(d[5]["w"]).cuda())
        self.conv16.bias = torch.nn.Parameter(torch.from_numpy(d[5]["b"]).cuda())
        self.relu16 = nn.ReLU(inplace=True)
        # 112 x 112

        self.reflecPad17 = nn.ReflectionPad2d((1,1,1,1))
        self.conv17 = nn.Conv2d(128,64,3,1,0)
        self.conv17.weight = torch.nn.Parameter(torch.from_numpy(d[6]["w"]).cuda())
        self.conv17.bias = torch.nn.Parameter(torch.from_numpy(d[6]["b"]).cuda())
        self.relu17 = nn.ReLU(inplace=True)
        # 112 x 112

        self.unpool3 = nn.UpsamplingNearest2d(scale_factor=2)
        # 224 x 224

        self.reflecPad18 = nn.ReflectionPad2d((1,1,1,1))
        self.conv18 = nn.Conv2d(64,64,3,1,0)
        self.conv18.weight = torch.nn.Parameter(torch.from_numpy(d[7]["w"]).cuda())
        self.conv18.bias = torch.nn.Parameter(torch.from_numpy(d[7]["b"]).cuda())
        self.relu18 = nn.ReLU(inplace=True)
        # 224 x 224

        self.reflecPad19 = nn.ReflectionPad2d((1,1,1,1))
        self.conv19 = nn.Conv2d(64,3,3,1,0)
        self.conv19.weight = torch.nn.Parameter(torch.from_numpy(d[8]["w"]).cuda())
        self.conv19.bias = torch.nn.Parameter(torch.from_numpy(d[8]["b"]).cuda())



    def forward(self,x):
        # decoder
        out = self.reflecPad11(x)
        out = self.conv11(out)
        out = self.relu11(out)
        out = self.unpool(out)
        out = self.reflecPad12(out)
        out = self.conv12(out)

        out = self.relu12(out)
        out = self.reflecPad13(out)
        out = self.conv13(out)
        out = self.relu13(out)
        out = self.reflecPad14(out)
        out = self.conv14(out)
        out = self.relu14(out)
        out = self.reflecPad15(out)
        out = self.conv15(out)
        out = self.relu15(out)
        out = self.unpool2(out)
        out = self.reflecPad16(out)
        out = self.conv16(out)
        out = self.relu16(out)
        out = self.reflecPad17(out)
        out = self.conv17(out)
        out = self.relu17(out)
        out = self.unpool3(out)
        out = self.reflecPad18(out)
        out = self.conv18(out)
        out = self.relu18(out)
        out = self.reflecPad19(out)
        out = self.conv19(out)
        return out


class level_5_decoder(nn.Module):
    def __init__(self):
        super(level_5_decoder,self).__init__()
        d = load_t7("models/feature_invertor_conv5_1.t7")
        # decoder
        self.reflecPad15 = nn.ReflectionPad2d((1,1,1,1))
        self.conv15 = nn.Conv2d(512,512,3,1,0)
        self.conv15.weight = torch.nn.Parameter(torch.from_numpy(d[0]["w"]).cuda())
        self.conv15.bias = torch.nn.Parameter(torch.from_numpy(d[0]["b"]).cuda())
        self.relu15 = nn.ReLU(inplace=True)

        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        # 28 x 28

        self.reflecPad16 = nn.ReflectionPad2d((1,1,1,1))
        self.conv16 = nn.Conv2d(512,512,3,1,0)
        self.conv16.weight = torch.nn.Parameter(torch.from_numpy(d[1]["w"]).cuda())
        self.conv16.bias = torch.nn.Parameter(torch.from_numpy(d[1]["b"]).cuda())
        self.relu16 = nn.ReLU(inplace=True)
        # 28 x 28

        self.reflecPad17 = nn.ReflectionPad2d((1,1,1,1))
        self.conv17 = nn.Conv2d(512,512,3,1,0)
        self.conv17.weight = torch.nn.Parameter(torch.from_numpy(d[2]["w"]).cuda())
        self.conv17.bias = torch.nn.Parameter(torch.from_numpy(d[2]["b"]).cuda())
        self.relu17 = nn.ReLU(inplace=True)
        # 28 x 28

        self.reflecPad18 = nn.ReflectionPad2d((1,1,1,1))
        self.conv18 = nn.Conv2d(512,512,3,1,0)
        self.conv18.weight = torch.nn.Parameter(torch.from_numpy(d[3]["w"]).cuda())
        self.conv18.bias = torch.nn.Parameter(torch.from_numpy(d[3]["b"]).cuda())
        self.relu18 = nn.ReLU(inplace=True)
        # 28 x 28

        self.reflecPad19 = nn.ReflectionPad2d((1,1,1,1))
        self.conv19 = nn.Conv2d(512,256,3,1,0)
        self.conv19.weight = torch.nn.Parameter(torch.from_numpy(d[4]["w"]).cuda())
        self.conv19.bias = torch.nn.Parameter(torch.from_numpy(d[4]["b"]).cuda())
        self.relu19 = nn.ReLU(inplace=True)
        # 28 x 28

        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
        # 56 x 56

        self.reflecPad20 = nn.ReflectionPad2d((1,1,1,1))
        self.conv20 = nn.Conv2d(256,256,3,1,0)
        self.conv20.weight = torch.nn.Parameter(torch.from_numpy(d[5]["w"]).cuda())
        self.conv20.bias = torch.nn.Parameter(torch.from_numpy(d[5]["b"]).cuda())
        self.relu20 = nn.ReLU(inplace=True)
        # 56 x 56

        self.reflecPad21 = nn.ReflectionPad2d((1,1,1,1))
        self.conv21 = nn.Conv2d(256,256,3,1,0)
        self.conv21.weight = torch.nn.Parameter(torch.from_numpy(d[6]["w"]).cuda())
        self.conv21.bias = torch.nn.Parameter(torch.from_numpy(d[6]["b"]).cuda())
        self.relu21 = nn.ReLU(inplace=True)

        self.reflecPad22 = nn.ReflectionPad2d((1,1,1,1))
        self.conv22 = nn.Conv2d(256,256,3,1,0)
        self.conv22.weight = torch.nn.Parameter(torch.from_numpy(d[7]["w"]).cuda())
        self.conv22.bias = torch.nn.Parameter(torch.from_numpy(d[7]["b"]).cuda())
        self.relu22 = nn.ReLU(inplace=True)

        self.reflecPad23 = nn.ReflectionPad2d((1,1,1,1))
        self.conv23 = nn.Conv2d(256,128,3,1,0)
        self.conv23.weight = torch.nn.Parameter(torch.from_numpy(d[8]["w"]).cuda())
        self.conv23.bias = torch.nn.Parameter(torch.from_numpy(d[8]["b"]).cuda())
        self.relu23 = nn.ReLU(inplace=True)

        self.unpool3 = nn.UpsamplingNearest2d(scale_factor=2)
        # 112 X 112

        self.reflecPad24 = nn.ReflectionPad2d((1,1,1,1))
        self.conv24 = nn.Conv2d(128,128,3,1,0)
        self.conv24.weight = torch.nn.Parameter(torch.from_numpy(d[9]["w"]).cuda())
        self.conv24.bias = torch.nn.Parameter(torch.from_numpy(d[9]["b"]).cuda())
        self.relu24 = nn.ReLU(inplace=True)

        self.reflecPad25 = nn.ReflectionPad2d((1,1,1,1))
        self.conv25 = nn.Conv2d(128,64,3,1,0)
        self.conv25.weight = torch.nn.Parameter(torch.from_numpy(d[10]["w"]).cuda())
        self.conv25.bias = torch.nn.Parameter(torch.from_numpy(d[10]["b"]).cuda())
        self.relu25 = nn.ReLU(inplace=True)

        self.unpool4 = nn.UpsamplingNearest2d(scale_factor=2)

        self.reflecPad26 = nn.ReflectionPad2d((1,1,1,1))
        self.conv26 = nn.Conv2d(64,64,3,1,0)
        self.conv26.weight = torch.nn.Parameter(torch.from_numpy(d[11]["w"]).cuda())
        self.conv26.bias = torch.nn.Parameter(torch.from_numpy(d[11]["b"]).cuda())
        self.relu26 = nn.ReLU(inplace=True)

        self.reflecPad27 = nn.ReflectionPad2d((1,1,1,1))
        self.conv27 = nn.Conv2d(64,3,3,1,0)
        self.conv27.weight = torch.nn.Parameter(torch.from_numpy(d[12]["w"]).cuda())
        self.conv27.bias = torch.nn.Parameter(torch.from_numpy(d[12]["b"]).cuda())

    def forward(self,x):
        # decoder
        out = self.reflecPad15(x)
        out = self.conv15(out)
        out = self.relu15(out)
        out = self.unpool(out)
        out = self.reflecPad16(out)
        out = self.conv16(out)
        out = self.relu16(out)
        out = self.reflecPad17(out)
        out = self.conv17(out)
        out = self.relu17(out)
        out = self.reflecPad18(out)
        out = self.conv18(out)
        out = self.relu18(out)
        out = self.reflecPad19(out)
        out = self.conv19(out)
        out = self.relu19(out)
        out = self.unpool2(out)
        out = self.reflecPad20(out)
        out = self.conv20(out)
        out = self.relu20(out)
        out = self.reflecPad21(out)
        out = self.conv21(out)
        out = self.relu21(out)
        out = self.reflecPad22(out)
        out = self.conv22(out)
        out = self.relu22(out)
        out = self.reflecPad23(out)
        out = self.conv23(out)
        out = self.relu23(out)
        out = self.unpool3(out)
        out = self.reflecPad24(out)
        out = self.conv24(out)
        out = self.relu24(out)
        out = self.reflecPad25(out)
        out = self.conv25(out)
        out = self.relu25(out)
        out = self.unpool4(out)
        out = self.reflecPad26(out)
        out = self.conv26(out)
        out = self.relu26(out)
        out = self.reflecPad27(out)
        out = self.conv27(out)
        return out
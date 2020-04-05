import json
import cv2
from encoder import style_features,\
    level_5_encoder, \
    level_4_encoder, \
    level_3_encoder, \
    level_2_encoder, \
    level_1_encoder

from decoder import \
    level_5_decoder, \
    level_4_decoder, \
    level_3_decoder, \
    level_2_decoder, \
    level_1_decoder

#from wct import wct
from utils import load_image
from wct import WCT
import torch
import torchvision.utils as vutils
import os

class Styletransfer:
    def __init__(self):
        self.h = 0
        self._style_settings = None
        self.net = None
        self.current_style = None
        self.style_vars = None
        self.level_5_encoder = level_5_encoder()
        self.level_5_decoder = level_5_decoder()
        self.level_4_encoder = level_4_encoder()
        self.level_4_decoder = level_4_decoder()
        self.level_3_encoder = level_3_encoder()
        self.level_3_decoder = level_3_decoder()
        self.level_2_encoder = level_2_encoder()
        self.level_2_decoder = level_2_decoder()
        self.level_1_encoder = level_1_encoder()
        self.level_1_decoder = level_1_decoder()
        self.WCT = WCT()

    def run(self,image,style,filename = "test.png"):
        self.set_style(style)
        img = load_image(image,size = self.current_style["content_size"])
        #print("level 5")

        for level,weight in json.loads(self.current_style["procedure"]):
            proc = self.style_level_1
            if level == 2:
                proc = self.style_level_2
            elif level == 3:
                proc = self.style_level_3
            elif level == 4:
                proc = self.style_level_4
            elif level == 5:
                proc = self.style_level_5
            img = proc(img,weight)
        vutils.save_image(img.data.cpu().float(), filename)

    def style_level_1(self,image,weight):
        style_params = self.style_vars["1"]
        print("level 1: encoding content features.")
        content_feature = self.level_1_encoder(image)
        print(content_feature.size())
        print("coloring")
        colored = self.WCT.content_coloring(
            content_feature,
            style_params,
            weight)
        print("reconstructing image")
        return self.level_1_decoder(colored)

    def style_level_2(self,image,weight):
        style_params = self.style_vars["2"]
        print("level 2: encoding content features.")
        content_feature = self.level_2_encoder(image)
        print(content_feature.size())
        print("coloring")
        colored = self.WCT.content_coloring(
            content_feature,
            style_params,
            weight)
        print("reconstructing image")
        return self.level_2_decoder(colored)

    def style_level_3(self,image,weight):
        style_params = self.style_vars["3"]
        print("level 3: encoding content features.")
        content_feature = self.level_3_encoder(image)
        print(content_feature.size())
        print("coloring")
        colored = self.WCT.content_coloring(
            content_feature,
            style_params,
            weight)
        print("reconstructing image")
        return self.level_3_decoder(colored)

    def style_level_4(self,image,weight):
        style_params = self.style_vars["4"]
        print("level 4: encoding content features.")
        content_feature = self.level_4_encoder(image)
        print(content_feature.size())
        print("coloring")
        colored = self.WCT.content_coloring(
            content_feature,
            style_params,
            weight)
        print("reconstructing image")
        return self.level_4_decoder(colored)

    def style_level_5(self,image,weight):
        style_params = self.style_vars["5"]
        print("level 5: encoding content features.")
        content_feature = self.level_5_encoder(image)
        print(content_feature.size())
        print("coloring")
        colored = self.WCT.content_coloring(
            content_feature,
            style_params,
            weight)
        print("reconstructing image")
        return self.level_5_decoder(colored)

    def set_style(self,style_key):
        if self.current_style != style_key:
            self.current_style = self.style_settings(style_key)
        if self.style_vars is None:
            print("loading style from disk")
            self.style_vars = torch.load("Resources/Styles/" + style_key + ".pt")
            print("done")

    def style_settings(self,key):
        if self._style_settings is None:
            self._style_settings = json.load(open("style_settings.json","r"))
        if (not "cache" in self._style_settings[key]) or not (os.path.isfile("Resources/Styles/" + key + ".pt")):
            self.make_style_cache(key)
        return self._style_settings[key]

    def make_style_cache(self,key):
        self.style_vars = None
        torch.cuda.empty_cache()
        print("loading image")
        image = load_image(self._style_settings[key]["image"],size = self._style_settings[key]["style_size"])
        print("getting style")
        net = style_features().cuda()
        net_output = net(image)
        ret = {}
        i = 1
        for style_feature in net_output:
            ret[str(i)] = self.WCT.style_params(style_feature)
            i += 1
        fname = "Resources/Styles/{name}.pt".format(name=key)
        torch.save(ret,fname)
        self._style_settings[key]["cache"] = fname
        json.dump(self._style_settings,open("style_settings.json","w"),indent = 1,sort_keys = True)
        del image
        del net
        torch.cuda.empty_cache()
        self.style_vars = ret

st = Styletransfer()
st.run("Resources/road.jpg","monet")

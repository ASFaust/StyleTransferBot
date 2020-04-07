import json
import cv2
import hashlib
from encoder import style_features,\
    level_5_encoder, \
    level_4_encoder, \
    level_3_encoder, \
    level_2_encoder, \
    level_1_encoder, \
    part_encoder_lvl1_2, \
    part_encoder_lvl2_3, \
    part_encoder_lvl3_4, \
    part_encoder_lvl4_5

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
import requests
import urllib.request


class Styletransfer:
    def __init__(self):
        self.h = 0
        self._style_settings = None
        self.net = None
        self.current_style = None
        self.style_vars = None
        self.current_cache_key = None
        self.encoder_cache = None
        self.decoder_cache = None
        self.last_level = -1

    def run(self,image,style,filename = "test.png",intermediates = False):
        self.WCT = WCT()
        self.set_style(style)
        img = load_image(image,size = self.current_style["content_size"])
        print(self.current_style["procedure"])
        i = 0
        for level,weight in json.loads(self.current_style["procedure"]):
            img = self.style(img,weight,level)
            if intermediates:
                print("---" * 20)
                ff = filename.split(".")
                fname =  ".".join(ff[:-1]) + "{:02d}.".format(i) + ff[-1]
                print("saving as " + fname)
                vutils.save_image(img.data.cpu().float(), fname)
            i += 1
        if not intermediates:
            print("---"*20)
            print("saving as " + filename)
            vutils.save_image(img.data.cpu().float(), filename)
        del self.WCT
        try:
            del self.encoder_cache
            del self.decoder_cache
        except:
            pass
        self.last_level = -1
        torch.cuda.empty_cache()

    def style(self,image,weight,level):
        print("loading encoders & decoders for level " + str(level))
        with torch.no_grad():
            if (self.last_level == level) or ((min(level,self.last_level) == 5) and (max(self.last_level,level) == 6)):
                encoder = self.encoder_cache
                decoder = self.decoder_cache
            else:
                try:
                    del self.encoder_cache
                    del self.decoder_cache
                    torch.cuda.empty_cache()
                except:
                    pass
                if level == 1:
                    encoder = level_1_encoder()
                    decoder = level_1_decoder()
                elif level == 2:
                    encoder = level_2_encoder()
                    decoder = level_2_decoder()
                elif level == 3:
                    encoder = level_3_encoder()
                    decoder = level_3_decoder()
                elif level == 4:
                    encoder = level_4_encoder()
                    decoder = level_4_decoder()
                elif level in [5,6]:
                    encoder = level_5_encoder()
                    decoder = level_5_decoder()
            level_str = str(level)
            if level == 6:
                osize = image.size()
                image = self.lvl6_downscale(image)
                level_str = "10"
            style_params = self.style_vars[level_str]
            print("encoding content features")
            content_feature = encoder(image)
            print(content_feature.size())
            print("coloring")
            colored = self.WCT.content_coloring(
                content_feature,
                style_params,
                weight)
            print("reconstructing image")
            ret = decoder(colored)
            print("ret size: " + str(ret.size()))
            self.encoder_cache = encoder
            self.decoder_cache = decoder
            self.last_level = level
        if level == 6:
            ret = self.lvl6_upscale(ret,osize)
        return ret

    def lvl6_downscale(self,image,scale = 224):
        i_size = image.size()
        if i_size[2] > i_size[3]:
            new_2 = int(i_size[2] / i_size[3] * scale)
            new_3 = scale
        else:
            new_3 = int(i_size[3] / i_size[2] * scale)
            new_2 = scale
        ret = torch.nn.functional.interpolate(image,size = (new_2,new_3),mode = "bilinear",align_corners = True)
        return ret

    def lvl6_upscale(self,image,osize):
        return torch.nn.functional.interpolate(image,size = (osize[2],osize[3]), mode = "bilinear",align_corners = True)

    def set_style(self,style_key):
        if self._style_settings is None:
            self._style_settings = json.load(open("style_settings.json","r"))
        self.current_style = self._style_settings[style_key]
        cache_key = self.get_style_cache_key(style_key)
        if not os.path.isfile(cache_key):
            print("baking style cache")
            self.make_style_cache(style_key,cache_key)
        elif self.current_cache_key != cache_key:
            print("loading style from disk")
            self.style_vars = torch.load(cache_key)
            print("done")
            self.current_cache_key = cache_key

    def get_style_cache_key(self,key):
        if self._style_settings is None:
            self._style_settings = json.load(open("style_settings.json","r"))
        style = self._style_settings[key]
        cache_key = str(hashlib.sha1((str(style["style_size"])+style["image"]).encode("utf-8")).hexdigest())
        return "Resources/Styles/cache/" + cache_key + ".pt"

    def make_style_cache(self,key,cache_key):
        self.style_vars = None
        torch.cuda.empty_cache()
        image = load_image(self._style_settings[key]["image"],size = self._style_settings[key]["style_size"])
        net = style_features().cuda()
        net_output = net(image)
        ret = {}
        i = 1
        for style_feature in net_output:
            ret[str(i)] = self.WCT.style_params(style_feature)
            i += 1
        #scale the input image accordingly to the smaller size
        ss = self._style_settings[key]
        image = self.lvl6_downscale(image,int(224/(ss["content_size"] / ss["style_size"])))
        net_output = net(image)
        for style_feature in net_output:
            ret[str(i)] = self.WCT.style_params(style_feature)
            print(i)
            i += 1
        torch.save(ret,cache_key)
        del image
        del net
        torch.cuda.empty_cache()
        self.style_vars = ret

def style_random_source(style):
    #
    image_info = requests.get(url = "https://www.shitpostbot.com/api/randsource").json()["sub"]
    title = image_info["name"]
    online_file = "https://www.shitpostbot.com/" + image_info["img"]["full"]
    local_file = "Resources/memes/" + online_file.split("/")[-1]
    print("downloading " + str(title) + " from " + online_file + " to " + local_file)
    urllib.request.urlretrieve(online_file,local_file)
    print("downloaded it.")
    styled_file = "Results/memes/" + ".".join(online_file.split("/")[-1].split(".")[:-1]) + ".png"
    st = Styletransfer()
    st.run(local_file, style, styled_file, intermediates=True)
    #st.run("Resources/bob1.jpg","what2","bob1.png")

styles = json.load(open("style_settings.json","r"))

for style in ["face","starry","wave"]:
    style_random_source(style)
#st.run("Resources/house1.jpg",style,"Results/house1.png",intermediates = True)
#st.run("Resources/road.jpg",style,"Results/road1.png",intermediates = True)
#st.run("Resources/bob1.jpg",style,"Results/bob1.png",intermediates = True)
#st.run("Resources/woody.jpg",style,"Results/woody.png",intermediates = True)
#st.explore_style("what2",style_size = 800,prefix = "[2,0.5],[5,1.0],[5,1.0]")
#st.style_settings("monet")


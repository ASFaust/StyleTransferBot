import json
import cv2
import hashlib
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

    def explore_style_0(self,style,style_size):
        print("exploring " + style)
        if self._style_settings is None:
            self._style_settings = json.load(open("style_settings.json","r"))
        for source in ["road.jpg","house1.jpg", "woody.jpg", "meme1.jpg","bob1.jpg"]:
            source_file = "Resources/" + source
            self._style_settings[style]["style_size"] = style_size
            for weight in [0.5, 0.75, 1.0]:
                self._style_settings[style]["procedure"] = "[[5,{w}],[4,{w}],[3,{w}],[2,{w}],[1,{w}]]".format(w = str(weight))
                result = "Results/full_" + style + "_" + source + "_size_" + str(style_size) + "_w_" + str(weight)  + ".png"
                self.run(source_file, style, result)

    def explore_style(self,style,style_size,prefix = "",weights = [0.5,0.75,1.0]):
        print("exploring " + style)
        if self._style_settings is None:
            self._style_settings = json.load(open("style_settings.json","r"))
        for source in ["road.jpg","house1.jpg", "woody.jpg", "meme1.jpg","bob1.jpg"]:
            source_file = "Resources/" + source
            self._style_settings[style]["style_size"] = style_size
            if prefix != "":
                procedure = "[" + prefix + "]"
                self._style_settings[style]["procedure"] = procedure
                result = "Results/" + style + "_" + source + "_size_" + str(style_size) + "_" + procedure + ".png"
                if not os.path.isfile(result):
                    self.run(source_file, style, result)
            for level in [1,2,3,4,5]:
                for weight in weights:
                    if prefix != ",":
                        procedure = "[" + prefix + ",[" + str(level) + "," + str(weight) + "]]"
                    else:
                        procedure = "[[" + str(level) + "," + str(weight) + "]]"
                    self._style_settings[style]["procedure"] = procedure
                    result = "Results/" + style + "_" + source + "_size_" + str(style_size) + "_" + procedure  + ".png"
                    if not os.path.isfile(result):
                        self.run(source_file,style,result)

    def style(self,image,weight,level):
        print("loading encoders & decoders for level " + str(level))
        if self.last_level == level:
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
            elif level == 5:
                encoder = level_5_encoder()
                decoder = level_5_decoder()
        style_params = self.style_vars[str(level)]
        print("encoding content features")
        iimage = torch.nn.functional.interpolate(image,size = (300,300),mode = 'bicubic')
        content_feature = encoder(iimage)
        print(content_feature.size())
        print("coloring")
        colored = self.WCT.content_coloring(
            content_feature,
            style_params,
            weight)
        print("reconstructing image")
        isize = image.size()
        csize = colored.size()
        new_size = [0,0]
        new_size[0] = (isize[2] / 300) * csize[2]
        new_size[1] = (isize[3] / 300) * csize[3]
        print("small colored size: " + str(csize))
        print("new colored size: " + str(new_size))
        icolored = torch.nn.functional.interpolate(content_feature,size = (int(new_size[0]),int(new_size[1])),mode = 'nearest')
        print("new colored size(): " + str(icolored.size()))
        ret = decoder(icolored)
        print("ret size: " + str(ret.size()))
        self.encoder_cache = encoder
        self.decoder_cache = decoder
        self.last_level = level
        return ret

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
        torch.save(ret,cache_key)
        del image
        del net
        torch.cuda.empty_cache()
        self.style_vars = ret

st = Styletransfer()
#st.run("Resources/bob1.jpg","what2","bob1.png")

#st.run("Resources/cage.jpg","fish","cage.png",intermediates = True)
#st.run("Resources/road.jpg","fish","road1.png",intermediates = True)
#st.run("Resources/bob1.jpg","fish","bob1.png",intermediates = True)
#st.run("Resources/woody.jpg","fish","woody.png",intermediates = True)
#st.run("Resources/woody.jpg","fish","woody.png",intermediates = True)
st.run("Resources/andi.jpg","fish","andi.png",intermediates = True)
#st.explore_style("what2",style_size = 800,prefix = "[2,0.5],[5,1.0],[5,1.0]")
#st.style_settings("monet")


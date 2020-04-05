from utils import load_t7
import numpy as np
import cv2



"""
#This little test shows us that the vgg weights are the same in all convs
#so that we only have to run it once to obtain the style :D
a = load_t7("models/vgg_normalised_conv5_1.t7")
b = load_t7("models/vgg_normalised_conv4_1.t7")

for i in range(len(b)):
    w1 = a[i]["w"]
    w2 = b[i]["w"]

    print("comparison: ")
    print(w1.shape)
    print(w2.shape)
    print(np.mean(w2-w1))
    
"""
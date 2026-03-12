import cv2
import numpy as np
import os
from tqdm import tqdm
list_np = os.listdir("/root/RXDistill/dataset/ade_mf_signal/rgbx_np/training/")

for i in tqdm(range(len(list_np))):
    img_rgbx = np.load("/root/RXDistill/dataset/ade_mf_signal/rgbx_np/training/"+list_np[i])
    for n in range(5):
        np.save("/root/RXDistill/dataset/ade_mf/rgbx_np/training/" +str(n)+"_"+ list_np[i], img_rgbx)

        
list_ann = os.listdir("/root/RXDistill/dataset/ade_mf_signal/annotations/training/")
for i in tqdm(range(len(list_ann))):
    img_ann = cv2.imread("/root/RXDistill/dataset/ade_mf_signal/annotations/training/"+list_ann[i],-1)
    for n in range(5):
        cv2.imwrite("/root/RXDistill/dataset/ade_mf/annotations/training/" +str(n)+"_"+ list_ann[i], img_ann)
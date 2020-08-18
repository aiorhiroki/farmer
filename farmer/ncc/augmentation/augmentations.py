# from keras.preprocessing import image
import numpy as np
import albumentations
from PIL import Image, ImageOps

class AutoContrast(albu.ImageOnlyTransform):
    """オートコントラスト調整
    画像のコントラストを最大化（平坦化）する
    入力画像のヒストグラムを計算し、
    cutoffで指定されたパーセント(< 50)以上の明るさ/暗さのピクセルを除去し、
    最も暗いピクセルが黒（0）に、最も明るいピクセルが白（255）になるようマッピングし直す
    """
    def __init__(self, cutoff=0, always_apply=False, p=0.5):
        super(AutoContrast, self).__init__(always_apply, p)
        self.cutoff = cutoff
    def apply(self, img, **params):
        img_p = Image.fromarray(img)
        autocon = ImageOps.autocontrast(img_p, self.cutoff)
        return np.asarray(autocon)

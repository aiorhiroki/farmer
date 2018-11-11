from farmer.Classifier import Classifier
import os

os.chdir('./examples/fit_from_directory/')

'''  mnist to image file from
from PIL import Image
import chainer

def save(data, index, num):
    img = Image.new("L", (28, 28))
    pix = img.load()
    for i in range(28):
        for j in range(28):
            pix[i, j] = int(data[i+j*28]*256)
    img2 = img.resize((28, 28))
    filename = str(num) + "/test" + "{0:04d}".format(index) + ".png"
    img2.save(filename)
    # print (filename)

_, test = chainer.datasets.get_mnist()
if os.path.isdir('mnist') is False:
    os.mkdir('mnist')
os.chdir('mnist')
for i in range(10):
    dirname = str(i)
    if os.path.isdir(dirname) is False:
        os.mkdir(dirname)
for i in range(len(test)):
    save(test[i][0], i, test[i][1])
to here '''


Classifier().fit_from_directory('mnist')

import colorsys
from .palette import palettes
import numpy as np
import os
import random
from PIL import Image
import cv2


def random_colors(N, bright=True, scale=True, shuffle=False):
    """ Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    if scale:
        colors = tuple(np.array(colors)*255)
    if shuffle:
        random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """ Apply the given mask to the image.
    image: (height, width, channel)
    mask: (height, width)
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def convert_to_palette(numpy_image):
    pil_palette = Image.fromarray(np.uint8(numpy_image), mode="P")
    pil_palette.putpalette(palettes)
    return pil_palette


def change_color_palettes(image_files, colors):
    for image_file in image_files:
        pil_img = Image.open(image_file)
        numpy_image = np.array(pil_img, dtype=np.uint8)
        for i, color in enumerate(colors):
            numpy_image[numpy_image == color] = i
        pil_palette = convert_to_palette(numpy_image)
        pil_palette.save(image_file)


def concat_images(im1, im2, palette, mode):
    if mode == "P":
        assert palette is not None
        dst = Image.new("P", (im1.width + im2.width, im1.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        dst.putpalette(palette)
    elif mode == "RGB":
        dst = Image.new("RGB", (im1.width + im2.width, im1.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
    else:
        raise NotImplementedError

    return dst


def cast_to_pil(ndarray, palette, index_void=None):
    # index_void: 境界線のindexで学習・可視化の際は背景色と同じにする。
    assert len(ndarray.shape) == 3
    res = np.argmax(ndarray, axis=2)
    if index_void is not None:
        res = np.where(res == index_void, 0, res)
    image = Image.fromarray(np.uint8(res), mode="P")
    image.putpalette(palette)
    return image

def generate_segmentation_result(
    nb_classes,
    height,
    width,
    annotations,
    model,
    save_dir,
    train_colors=None,
):
    image_util = ImageUtil(nb_classes, (height, width))
    for sample_image_path in annotations:
        sample_image = image_util.read_image(
            sample_image_path[0],
            anti_alias=True
        )
        sample_image = np.asarray(sample_image, dtype=np.float32)
        segmented = image_util.read_image(
            sample_image_path[1],
            normalization=False,
            train_colors=train_colors
        )
        segmented = image_util.cast_to_onehot(segmented)
        output = model.predict(np.expand_dims(sample_image, axis=0))

        result_image = get_imageset(sample_image, output[0], segmented)
        save_image_name = os.path.basename(sample_image_path[0])
        result_image.save(f"{save_dir}/{save_image_name}")

def get_imageset(
    image_in_np,
    image_out_np,
    image_gt_np,
    palette=palettes,
    index_void=None
):
    # 3つの画像(in, out, gt)をくっつけます。
    image_out = cast_to_pil(
        image_out_np, palette, index_void
    )
    image_tc = cast_to_pil(
        image_gt_np, palette, index_void
    )
    image_merged = concat_images(
        image_out, image_tc, palette, "P"
    ).convert("RGB")
    image_in_pil = Image.fromarray(
        np.uint8(image_in_np * 255), mode="RGB"
    )
    image_result = concat_images(
        image_in_pil, image_merged, None, "RGB"
    )
    return image_result


class ImageUtil:

    def __init__(
        self,
        nb_classes: int,
        size: (int, int)
    ):
        self.nb_classes = nb_classes
        self.size = size[::-1]
        self.current_raw_size = None
        self.current_raw_frame = None

    def read_image(
        self,
        file_path: str,
        normalization=True,
        anti_alias=False,
        train_colors=None
    ):
        image = Image.open(file_path)
        self.current_raw_frame = image
        self.current_raw_size = image.size
        if self.size != self.current_raw_size:
            resample = Image.LANCZOS if anti_alias else Image.NEAREST
            image = image.resize(self.size, resample)
        # delete alpha channel
        if image.mode == "RGBA":
            image = image.convert("RGB")
        image = np.asarray(image)
        if normalization:
            image = image / 255.0
        if train_colors:
            image = self._convert_colors(image, train_colors)

        return image

    def cast_to_onehot(
        self,
        labels: list
    ):
        labels = np.asarray(labels, dtype=np.uint8)
        # Classification
        if len(labels.shape) == 1:
            one_hot = np.eye(self.nb_classes)
        # Segmentation
        else:
            one_hot = np.identity(self.nb_classes)
        return one_hot[labels]

    def _cast_to_frame(
        self,
        prediction,
        size
    ):
        res = np.argmax(prediction, axis=2)
        image = Image.fromarray(np.uint8(res), mode="P")
        image.putpalette(palettes)
        image = image.resize(self.current_raw_size, Image.LANCZOS)
        image = image.convert("RGB")
        return np.asarray(np.asarray(image)*255, dtype=np.uint8)

    def blend_image(
        self,
        output_image,
        size
    ):
        input_frame = np.array(self.current_raw_frame, dtype=np.uint8)
        output_frame = self._cast_to_frame(output_image, size)
        blended = cv2.addWeighted(
            src1=input_frame,
            src2=output_frame,
            alpha=0.7,
            beta=0.9,
            gamma=2.2
        )
        return cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)

    def _convert_colors(self, label_gray, train_colors):
        label = np.zeros(label_gray.shape)
        for train_id, train_color in enumerate(train_colors):
            label[label_gray == train_color] = train_id + 1
        return label

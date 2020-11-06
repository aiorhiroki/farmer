from typing import Tuple
import colorsys
from .palette import palettes
import numpy as np
import random
from PIL import Image, ImageDraw
import cv2


def random_colors(N, bright=True, scale=True, shuffle=False):
    """ Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    if scale:
        colors = tuple(np.array(colors) * 255)
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


def draw_text(
    image,
    put_text,
    top_left,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    scale=1,
    thickness=1,
    line_type=cv2.LINE_8,
    color=(255, 255, 255)
):
    (_, text_height), _ = cv2.getTextSize(put_text, font, scale, thickness)
    bottom_left = (top_left[0], top_left[1] + text_height)
    cv2.putText(image, put_text, bottom_left, font, scale,
                color, thickness, line_type)


def get_labels(mask):
    hist = np.bincount(mask.flatten())
    return [i for i, n in enumerate(hist) if i > 0 and n > 0]


def fill_indexed(
    image,
    mask,
    palette,
    index_void=None
):
    for i in get_labels(mask):
        if i != index_void:
            image[mask==i] = palette[i]


def get_imageset(
    image_in_np,
    image_out_np,
    image_gt_np,
    palette=palettes,
    index_void=None,
    put_text=None
):
    image_in = np.uint8(image_in_np * 255)
    image_out = np.uint8(np.argmax(image_out_np, axis=2))
    image_gt = np.uint8(np.argmax(image_gt_np, axis=2))

    palette = np.reshape(palette, (-1, 3))

    rows, cols, channels = image_in.shape
    image_result = np.zeros((rows, cols * 3, channels), dtype=np.uint8)
    image_result[:,:cols] = image_in

    fill_indexed(image_result[:,cols:cols*2], image_out, palette, index_void)
    fill_indexed(image_result[:,cols*2:], image_gt, palette, index_void)

    if put_text is not None:
        draw_text(image_result, put_text, (cols, 0), scale=0.5)

    return image_result


class ImageUtil:

    def __init__(
        self,
        nb_classes: int,
        size: Tuple[int, int]
    ):
        self.nb_classes = nb_classes
        self.size = size[::-1]  # width, height
        self.current_raw_size = None
        self.current_raw_frame = None

    def read_image(
        self,
        file_path: str,
        train_colors=None
    ):
        image = Image.open(file_path)
        self.current_raw_frame = image
        self.current_raw_size = image.size

        # delete alpha channel
        if image.mode == "RGBA":
            image = image.convert("RGB")
        image = np.asarray(image)

        # re-mapping class index
        if train_colors:
            image = self._convert_colors(image, train_colors)

        return image

    def resize(
        self,
        image: np.ndarray,
        anti_alias=False
    ):
        image = Image.fromarray(np.uint8(image))
        resample = Image.LANCZOS if anti_alias else Image.NEAREST
        image = image.resize(self.size, resample)
        return np.asarray(image)

    def normalization(
        self,
        image: np.ndarray,
    ):
        return image / 255.0

    def cast_to_onehot(
        self,
        label: np.ndarray
    ):
        one_hot = np.identity(self.nb_classes)
        return one_hot[label]

    def _cast_to_frame(
        self,
        prediction,
        size,
    ):
        res = np.argmax(prediction, axis=2)
        image = Image.fromarray(np.uint8(res), mode="P")
        image.putpalette(palettes)
        image = image.resize(size, Image.LANCZOS)
        image = image.convert("RGB")
        return np.asarray(np.asarray(image) * 255, dtype=np.uint8)

    def blend_image(
        self,
        output_image,
        size,
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
            if type(train_color) == int:
                if train_color == 0 and train_id == 0:
                    continue
                label[label_gray == train_color] = train_id
            elif type(train_color) == dict:
                before_color, after_color = list(train_color.items())[0]
                label[label_gray == before_color] = after_color
        return label

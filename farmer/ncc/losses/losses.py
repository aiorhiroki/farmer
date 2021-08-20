import tensorflow as tf
import segmentation_models
from segmentation_models.base import Loss
from segmentation_models.losses import CategoricalCELoss
from ..losses import functional as F

segmentation_models.set_framework('tf.keras')


class DiceLoss(Loss):
    def __init__(self, beta=1, class_weights=None, flooding_level=0.):
        super().__init__(name='dice_loss')
        self.beta = beta
        self.class_weights = class_weights if class_weights is not None else 1
        self.flooding_level = flooding_level

    def __call__(self, gt, pr):
        return F.flooding(F.dice_loss(
            gt=gt,
            pr=pr,
            beta=self.beta,
            class_weights=self.class_weights
        ), self.flooding_level)


class JaccardLoss(Loss):
    def __init__(self, class_weights=None, flooding_level=0.):
        super().__init__(name='jaccard_loss')
        self.class_weights = class_weights if class_weights is not None else 1
        self.flooding_level = flooding_level

    def __call__(self, gt, pr):
        return F.flooding(F.jaccard_loss(
            gt=gt,
            pr=pr,
            class_weights=self.class_weights
        ), self.flooding_level)


class TverskyLoss(Loss):
    def __init__(self, alpha=0.45, beta=0.55, class_weights=None, flooding_level=0.):
        super().__init__(name='tversky_loss')
        self.alpha = alpha
        self.beta = beta
        self.class_weights = class_weights if class_weights is not None else 1.
        self.flooding_level = flooding_level

    def __call__(self, gt, pr):
        return F.flooding(F.tversky_loss(
            gt=gt,
            pr=pr,
            alpha=self.alpha,
            beta=self.beta,
            class_weights=self.class_weights
        ), self.flooding_level)


class FocalTverskyLoss(Loss):
    def __init__(self, alpha=0.45, beta=0.55, gamma=2.5, class_weights=None, flooding_level=0.):
        super().__init__(name='focal_tversky_loss')
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.class_weights = class_weights if class_weights is not None else 1.
        self.flooding_level = flooding_level

    def __call__(self, gt, pr):
        return F.flooding(F.focal_tversky_loss(
            gt=gt,
            pr=pr,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            class_weights=self.class_weights
        ), self.flooding_level)


class CategoricalFocalLoss(Loss):
    def __init__(self, alpha=0.25, gamma=2., class_weights=None, flooding_level=0.):
        super().__init__(name='categorical_focal_loss')
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights if class_weights is not None else 1.
        self.flooding_level = flooding_level

    def __call__(self, gt, pr):
        return F.flooding(F.categorical_focal_loss(
            gt,
            pr,
            alpha=self.alpha,
            gamma=self.gamma,
            class_weights=self.class_weights
        ), self.flooding_level)


class LogCoshDiceLoss(Loss):
    def __init__(self, beta=1, class_weights=None, flooding_level=0.):
        super().__init__(name='log_cosh_dice_loss')
        self.beta = beta
        self.class_weights = class_weights if class_weights is not None else 1
        self.flooding_level = flooding_level

    def __call__(self, gt, pr):
        return F.flooding(F.log_cosh_dice_loss(
            gt=gt,
            pr=pr,
            beta=self.beta,
            class_weights=self.class_weights
        ), self.flooding_level)


class LogCoshTverskyLoss(Loss):
    def __init__(self, alpha=0.3, beta=0.7, class_weights=None, flooding_level=0.):
        super().__init__(name='log_cosh_tversky_loss')
        self.alpha = alpha
        self.beta = beta
        self.class_weights = class_weights if class_weights is not None else 1.
        self.flooding_level = flooding_level

    def __call__(self, gt, pr):
        return F.flooding(F.log_cosh_tversky_loss(
            gt=gt,
            pr=pr,
            alpha=self.alpha,
            beta=self.beta,
            class_weights=self.class_weights
        ), self.flooding_level)


class LogCoshFocalTverskyLoss(Loss):
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.3, class_weights=None, flooding_level=0.):
        super().__init__(name='log_cosh_focal_tversky_loss')
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.class_weights = class_weights if class_weights is not None else 1.
        self.flooding_level = flooding_level

    def __call__(self, gt, pr):
        return F.flooding(F.log_cosh_focal_tversky_loss(
            gt=gt,
            pr=pr,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            class_weights=self.class_weights
        ), self.flooding_level)


class LogCoshLoss(Loss):
    def __init__(self, base_loss, flooding_level=0., **kwargs):
        super().__init__(name=f'log_cosh_{base_loss}')
        self.loss = getattr(F, base_loss)
        self.flooding_level = flooding_level
        self.kwargs = kwargs

    def __call__(self, gt, pr):
        x = self.loss(gt, pr, **self.kwargs)
        return F.flooding(
            tf.math.log((tf.exp(x) + tf.exp(-x)) / 2.0),
            self.flooding_level)


class RelativeVolumeDifferenceLoss(Loss):
    def __init__(self, class_weights=None, flooding_level=0., **kwargs):
        super().__init__(name='rvd_loss')
        self.class_weights = class_weights if class_weights is not None else 1
        self.flooding_level = flooding_level

    def __call__(self, gt, pr):
        return F.flooding(F.rvd_loss(
            gt=gt,
            pr=pr,
            class_weights=self.class_weights
        ), self.flooding_level)


class BoundaryLoss(Loss):
    def __init__(self, flooding_level=0., **kwargs):
        super().__init__(name='boundary_loss')
        self.flooding_level = flooding_level

    def __call__(self, gt, pr):
        return F.flooding(F.surface_loss(
            gt=gt,
            pr=pr,
        ), self.flooding_level)

import segmentation_models
from segmentation_models.base import Loss
from segmentation_models.losses import (
    DiceLoss, JaccardLoss, CategoricalFocalLoss, CategoricalCELoss
)


segmentation_models.set_framework('tf.keras')


def dice_loss(beta=1, class_weights=None, class_indexes=None, **kwargs):
    loss = DiceLoss(
        beta=beta,
        class_weights=class_weights,
        class_indexes=class_indexes
    )
    return loss


def jaccard_loss(class_weights=None, class_indexes=None, per_image=False, **kwargs):
    loss = JaccardLoss(
        class_weights=class_weights,
        class_indexes=class_indexes,
        per_image=per_image
    )
    return loss


def categorical_focal_loss(alpha=0.25, gamma=2., class_indexes=None, **kwargs):
    loss = CategoricalFocalLoss(
        alpha=alpha,
        gamma=gamma,
        class_indexes=class_indexes
    )
    return loss


def categorical_crossentropy_loss(class_weights=None, class_indexes=None, **kwargs):
    loss = CategoricalCELoss(
        class_weights=class_weights,
        class_indexes=class_indexes
    )
    return loss


def cce_dice_loss(beta=1, class_weights=None, class_indexes=None, **kwargs):
    cce = categorical_crossentropy(
        class_weights=class_weights,
        class_indexes=class_indexes
    )
    dl = dice_loss(
        beta=beta,
        class_weights=class_weights,
        class_indexes=class_indexes
    )
    return cce + dl


def cce_jaccard_loss(class_weights=None, class_indexes=None, per_image=False, **kwargs):
    cce = categorical_crossentropy(
        class_weights=class_weights,
        class_indexes=class_indexes
    )
    jl = jaccard_loss(
        class_weights=class_weights,
        class_indexes=class_indexes,
        per_image=per_image
    )
    return cce + jl


def categorical_focal_dice_loss(alpha=0.25, beta=1, gamma=2.,
                                class_weights=None, class_indexes=None, **kwargs):
    cfl = categorical_focal_loss(
        alpha=alpha,
        gamma=gamma,
        class_indexes=class_indexes
    )
    dl = dice_loss(
        beta=beta,
        class_weights=class_weights,
        class_indexes=class_indexes
    )
    return cfl + dl


def categorical_focal_jaccard_loss(alpha=0.25, gamma=2., class_weights=None,
                                   class_indexes=None, per_image=False, **kwargs):
    cfl = categorical_focal_loss(
        alpha=alpha,
        gamma=gamma,
        class_indexes=class_indexes
    )
    jl = jaccard_loss(
        class_weights=class_weights,
        class_indexes=class_indexes,
        per_image=per_image
    )
    return cfl + jl

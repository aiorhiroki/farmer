from farmer import ncc


class EvalSegmentationTask:
    def __init__(self, config):
        self.config = config

    def command(self, annotation_set, model):
        eval_report = self._do_segmentation_evaluation_task(
            annotation_set, model
        )
        return eval_report

    def _do_segmentation_evaluation_task(self, annotation_set, model):
        iou_dice = ncc.metrics.iou_dice_val(
            self.config.nb_classes,
            self.config.height,
            self.config.width,
            annotation_set,
            model,
            self.config.train_colors
        )
        return iou_dice

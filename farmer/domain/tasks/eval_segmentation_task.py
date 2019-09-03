import ncc


class EvalSegmentationTask:

    def __init__(self, config):
        self.config = config

    def command(self, annotation_set, model):
        return self._do_segmentation_evaluation(annotation_set, model)

    def _do_segmentation_evaluation(self, annotation_set, model):
        iou = ncc.metrcs.iou_validation(
            self.config.nb_classes,
            self.config.height,
            self.config.width,
            annotation_set,
            model
        )
        return iou

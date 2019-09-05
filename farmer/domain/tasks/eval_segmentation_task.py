import ncc


class EvalSegmentationTask:

    def __init__(self, config):
        self.config = config

    def command(self, annotation_set, model):
        eval_report = self._do_segmentation_evaluation(annotation_set, model)
        return eval_report

    def _do_segmentation_evaluation_task(
        self,
        annotation_set,
        model
    ):
        iou = ncc.metrcs.iou_validation(
            self.config.nb_classes,
            self.config.height,
            self.config.width,
            annotation_set,
            model
        )
        return iou

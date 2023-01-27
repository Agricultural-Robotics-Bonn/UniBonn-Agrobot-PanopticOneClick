from modules.evaluator_base import EvaluatorBase
from .panoptic_quality import PanopticQuality

import torch

class PanopticEvaluator(EvaluatorBase):
  '''
  Panoptic Qualitity evaluator

  Expects panoptic mask and labels in the network output dictionary, i.e.:
    outputs['preds']['panoptic']
    outputs['labels']['panoptic']

  This are usually not provided by panoptic networks by default and
  panoptic output post-processing modules are needed such as
  `tag_seg_to_panoptic_postprocess.py`

  These can be added by populating the `post_process` member in the calling
  `thunder_module.py`
  '''
  def __init__(self, *args, **kwargs):
    super(PanopticEvaluator, self).__init__(*args, **kwargs)
    ## Populate metrics
    self.metric_dict['PQ'] = PanopticQuality(things=set(self.things_ids), stuff=set(self.stuff_ids),
                                             allow_unknown_preds_category=True)
    ## Populate monotonously increasing criterion
    self.best_criterions={'PQ':max}
    ## Populate Progress Bar metrics
    self.pbar_metrics=['val_epoch_PQ']

  def condition_net_output(self, outputs, loss=None):
    '''
    get panoptic predictions and labels from output dictionary

    Expected preds/labels shape:
      [batch, (sem_class, inst_IDs), h, w]

    returns tuple(torch.Tensor, torch.Tensor)
    '''
    if 'panoptic' not in outputs['preds'].keys():
      raise KeyError('Panoptic predictions not found in output dictionary, "panoptic" key not found')
    if 'panoptic' not in outputs['labels'].keys():
      raise KeyError('Panoptic labels not found in output dictionary, "panoptic" key not found')

    panoptic_preds = torch.clone( outputs['preds']['panoptic'] )
    panoptic_labels = torch.clone( outputs['labels']['panoptic'] )

    return panoptic_preds, panoptic_labels


  def compute(self):
    '''
    Compute all class-weighted metrics from incremental gathered batch results
    '''
    ret = {}
    for name, m in self.metric_dict.items():
      stat_classes = m.compute()
      # {metric_name : [Weighted average metric, class metric]
      # ret[name] = [self.weightedAVG(stat_classes), stat_classes]
      ret['all_PQ'] = stat_classes['all']['pq']
      ret['all_RQ'] = stat_classes['all']['rq']
      ret['all_SQ'] = stat_classes['all']['sq']
      ret['things_PQ'] = stat_classes['things']['pq']
      ret['things_RQ'] = stat_classes['things']['rq']
      ret['things_SQ'] = stat_classes['things']['sq']
      ret['stuff_PQ'] = stat_classes['stuff']['pq']
      ret['stuff_RQ'] = stat_classes['stuff']['rq']
      ret['stuff_SQ'] = stat_classes['stuff']['sq']

    return ret

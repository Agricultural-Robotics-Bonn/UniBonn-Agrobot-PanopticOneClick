from modules.evaluator_base import EvaluatorBase
import torchmetrics as metrics
import torch

class PanopticWeaksegEvaluator(EvaluatorBase):
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
    super().__init__(*args, **kwargs)
    ## Populate metrics
    self.metric_dict['iou'] = metrics.IoU(self.class_num, reduction='none')
    ## Populate monotonously increasing criterion
    self.best_criterions={'iou':max}
    ## Populate Progress Bar metrics
    self.pbar_metrics=['val_epoch_iou_plant']

  def condition_net_output(self, outputs, loss=None):
    '''
    get panoptic predictions and labels from output dictionary

    Expected preds/labels shape:
      [batch, (sem_class, inst_IDs), h, w]

    returns tuple(torch.Tensor, torch.Tensor)
    '''
    if 'panoptic' not in outputs['preds'].keys():
      raise KeyError('Panoptic predictions not found in output dictionary, "panoptic" key not found')

    panoptic_preds = outputs['preds']['panoptic']  # those aren't exacly panoptic but instance based on panoptic (using process_panoptic_weak)
    panoptic_targs = outputs['labels']['panoptic'][:,1,:,:]

    device = panoptic_preds.device

    h = panoptic_preds[0].size()[0]
    w = panoptic_preds[0].size()[1]

    ## loop batch
    for i, ( img_pred, img_targ ) in enumerate( zip( panoptic_preds, panoptic_targs ) ):

      ## collect instances
      weak_preds = torch.zeros( (len( torch.unique( img_targ ) )-1) , h, w).to( torch.int64 ).to( device )
      weak_targs = torch.zeros( (len( torch.unique( img_targ ) )-1) , h, w).to( torch.int64 ).to( device )

      skipped_0 = 0
      for ii, u in enumerate( torch.unique( img_targ )):
        if u == 0:
          skipped_0 = 1
          continue
        weak_preds[ii-skipped_0,...] = torch.where( img_pred==u, 1, 0 )
        weak_targs[ii-skipped_0,...] = torch.where( img_targ==u, 1, 0 )

      ## collect batches
      if i==0:
        weak_preds_combined = weak_preds
        weak_targs_combined = weak_targs
      else:
        weak_preds_combined = torch.cat( (weak_preds_combined, weak_preds), dim=0 )
        weak_targs_combined = torch.cat( (weak_targs_combined, weak_targs), dim=0 )

    return weak_preds_combined, weak_targs_combined



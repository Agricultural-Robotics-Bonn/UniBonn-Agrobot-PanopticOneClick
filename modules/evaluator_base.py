# -*- coding: utf-8 -*-
import itertools

import matplotlib.pyplot as plt
import numpy as np


import torch
from torch import nn

from pytorch_lightning.callbacks import Callback

def tb_confusion_matrix_img(cm, class_names, epsilon=1e-6):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """
    if isinstance(cm, torch.Tensor):
      cm = cm.cpu().numpy()

    fig = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis]+epsilon), decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Draw figure on canvas
    fig.canvas.draw()

    # Convert the figure to numpy array, read the pixel values and reshape the array
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # Normalize into 0-1 range for TensorBoard(X)
    img = img / 255.0
    # Swap axes (Newer API>=1.8 expects colors in first dim)
    img = np.swapaxes(img, 0, 2) # if your TensorFlow + TensorBoard version are >= 1.8
    img = np.swapaxes(img, 1, 2) # if your TensorFlow + TensorBoard version are >= 1.8
    plt.close(fig)

    return img


class EvaluatorStageWrapper:
  def __init__(self, evaluator=None, stage=None):
    if not evaluator or not isinstance(evaluator, nn.Module):
      raise TypeError('Invalid evaluator passed to evaluator callback class')
    self.eval = evaluator
    if any([stage == s for s in ['train', 'val', 'test']]):
      self.stage = stage
    else:
      raise ValueError('Unable to instanciate evaluator callback. Stage parameter must be one of: [train, val, test]')

  def setup(self, stage=None):
    #reset all metrics
    if any([stage == s for s in ['train', 'val', 'test']]):
      self.stage = stage
    self.eval.reset()

  def on_batch_end(self, pl_module, outputs, batch_idx):
    self.eval.log_metrics_batch(pl_module, outputs, batch_idx, self.stage)

  def on_epoch_end(self, pl_module):
    self.eval.log_metrics_epoch(pl_module, self.stage)
    self.eval.reset()


class EvaluatorTrainCallbacks(Callback):
  def __init__(self, evaluator=None):
    self.eval = EvaluatorStageWrapper(evaluator, 'train')
  # Train Callbacks
  def setup(self, trainer, pl_module, stage=None):
    self.eval.setup()
  def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
    self.eval.on_batch_end(pl_module, outputs, batch_idx)
  def on_train_epoch_end(self, trainer, pl_module):
    self.eval.on_epoch_end(pl_module)


class EvaluatorValCallbacks(Callback):
  def __init__(self, evaluator=None):
    self.eval = EvaluatorStageWrapper(evaluator, 'val')
  # Val Callbacks
  def setup(self, trainer, pl_module, stage=None):
    self.eval.setup()
  def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
    self.eval.on_batch_end(pl_module, outputs, batch_idx)
  def on_validation_epoch_end(self, trainer, pl_module):
    self.eval.on_epoch_end(pl_module)


class EvaluatorTestCallbacks(Callback):
  def __init__(self, evaluator=None):
    self.eval = EvaluatorStageWrapper(evaluator, 'test')
  # Test Callbacks
  def setup(self, trainer, pl_module, stage=None):
    self.eval.setup()
  def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
    self.eval.on_batch_end(pl_module, outputs, batch_idx)
  def on_test_epoch_end(self, trainer, pl_module):
    self.eval.on_epoch_end(pl_module)


class EvaluatorBase(nn.Module):
  def __init__(self, cfg={}, class_num=0, class_labels=[], class_weights=[],
               things_ids=[], stuff_ids=[],
               log_batch=True, log_epoch=True, log_best_metrics=True):
    """ Evaluator base class inteneded to be wrapped with a stage callback interface, e.g.: EvaluatorTrainCallbacks

    Manages torchmetric objects throughout an experiment and logs result in configured loggers.
    Takes typical evaluation parameters for most task either through arguments of by parsing a config dictionary

    Concrete Evaluators extending this class should populate:

      self.metric_dict: dictionary of metric objects to track troughout the experiment
      e.g.: self.metric_dict['F1'] = torchmetrics.F1(self.class_num, average='none')

      self.best_criterions: Dictionary of criterions to use for each best metric to be logged
      e.g.: self.best_criterions={'F1':max,
                                  'loss':min}

      self.pbar_metrics: Metrics to be reported in the procress bar
      e.g.: self.pbar_metrics=['val_epoch_iou_wavg', 'val_epoch_loss']

    Args:
        cfg (dict, optional): Defaults to {}.
          Parsed parameters:
            cfg['dataset']['class_num']
            cfg['dataset']['class_labels']
            cfg['dataset']['class_weights']
            cfg['dataset']['things_ids']
            cfg['dataset']['stuff_ids']

        class_num (int, optional): number of N evaluated classes. Defaults to 0.
        class_labels (list, optional): Classes label names, must be of lenght N. Defaults to [].
        class_weights (list, optional): Class weights for weifhted metrics, must be of lenght N. Defaults to [].
        things_ids (list, optional): Class IDs of things instances (used for panoptic segmentation). Defaults to [].
        stuff_ids (list, optional): Class IDs of stuff (used for panoptic segmentation). Defaults to [].
        log_batch (bool, optional): Flag to log metrics at the end of each batch. Defaults to True.
        log_epoch (bool, optional): Flag to log metrics at the end of each epoch. Defaults to True.
        log_best_metrics (bool, optional): Flag to log an extra monotonous version of a metric. Defaults to True.
    """

    super(EvaluatorBase, self).__init__()
    self.cfg = cfg
    dataset_cfg = cfg['dataset']

    # Get generic dataset parameters from the config dictionary if specified
    # otherwise use passed arguments
    self.class_labels = dataset_cfg['class_labels'] if 'class_labels' in dataset_cfg.keys() else class_labels
    assert(isinstance(self.class_labels, list))

    self.class_num = dataset_cfg['class_num'] if 'class_num' in dataset_cfg.keys() else class_num
    assert(isinstance(self.class_num, int))

    class_weights = dataset_cfg['class_weights'] if 'class_weights' in dataset_cfg.keys() else class_weights
    assert(isinstance(class_weights, list))
    if class_weights:
      class_weights = torch.Tensor(class_weights)
    else:
      class_weights = torch.ones(self.class_num)
    self.register_buffer('class_weights', class_weights, persistent=False)

    self.things_ids = dataset_cfg['things_ids'] if 'things_ids' in dataset_cfg.keys() else things_ids
    assert(isinstance(self.things_ids, list))

    self.stuff_ids = dataset_cfg['stuff_ids'] if 'stuff_ids' in dataset_cfg.keys() else stuff_ids
    assert(isinstance(self.stuff_ids, list))


    self.log_best_metrics = log_best_metrics
    self.best_metrics={}

    self.log_periods = []
    self.log_batch = log_batch
    if self.log_batch: self.log_periods.append('batch')
    self.log_epoch = log_epoch
    if self.log_batch: self.log_periods.append('epoch')


    ## Members to populate by concrete evaluator clases
    #####################################################
    self.metric_dict = nn.ModuleDict()
    self.best_criterions={}
    self.pbar_metrics = []


  def get_metric_names(self):
    metric_names = []
    for period in self.log_periods:
      for stat_name, stat in self.metric_dict.items():
        # all other metric names
        if stat_name == 'conf_matrix':
          metric_names.append(f'{period}_{stat_name}')
          continue
        # average metric names
        avg_str = 'wavg'
        if torch.all(self.class_weights == 1):
          avg_str = 'avg'
        metric_names.append(f'{period}_{stat_name}_{avg_str}')
        # Class-wise metric names
        for label in self.class_labels:
          metric_names.append(f'{period}_{stat_name}_{label}')

    return metric_names

  def reset(self):
    for m in self.metric_dict.values():
      m.reset()

  def condition_net_output(self, outputs, loss=None):
    '''
    Conditions the predictions and labels from network output

    This function should be overriden by concrete evaluation child clases
    if additional output post-processing is needed

    Ideally this funtions performs minimal post processing, otherwise
    using/writing a post-processing module is reccomended, extending
    `PostProcessBase`

    Must return a tuple (preds, labels) appropriate to be processed by
    all metrics in `self.metric_dict`
    '''
    return outputs['preds']['out'], outputs['labels']

  def update(self, outputs, loss=None):
    '''
    Incrementally update metric with predictions and labels

    This function should be overriden if additional evaluation inputs/capabilities
    are required by concrete evaluation child clases

    Must return a dict {'metric_name':metric_value}
      Ideally sores te updates of metrics in self.metric_dict
    '''
    preds, labels = self.condition_net_output(outputs, loss)
    # print( torch.unique( preds ), torch.unique( labels ) ); #import sys; sys.exit( 1 )
    return {name:m(preds, labels) for name, m in self.metric_dict.items()}

  def weightedAVG(self, class_scores):
    '''
    Compute class-wise weighted average score using weights specified in config file.
    If no weights are specified, the arithmetic mean is computed.
    '''
    return  ((class_scores * self.class_weights).sum() / self.class_weights.sum())

  def compute(self):
    '''
    Compute all class-weighted metrics from incremental gathered batch results
    '''
    ret = {}
    for name, m in self.metric_dict.items():
      stat_classes = m.compute()
      ret[name] = [self.weightedAVG(stat_classes), stat_classes]
    return ret

  def log_metrics_batch(self, pl_module, outputs, batch_idx, stage='', loss=None):
    if stage == 'train' and 'extra' in outputs.keys():
      outputs = outputs['extra']
    if any([isinstance(v, (list, tuple)) for k,v in outputs.items()]):
      out_keys = list(outputs.keys())
      seq_length = len(outputs[out_keys[0]])

      for i in range(seq_length):
        out = {k:outputs[k][i] for k in out_keys}
        self.log_metrics_single_batch(pl_module, out, batch_idx, stage, loss)

    else:
      self.log_metrics_single_batch(pl_module, outputs, batch_idx, stage, loss)

  def log_metrics_single_batch(self, pl_module, outputs, batch_idx, stage='', loss=None):
    # retrieve extra outputs to compute training metrics
    if stage == 'train' and 'extra' in outputs.keys():
      outputs = outputs['extra']
    # update metrics with current batch results
    stats_dict = self.update(outputs, loss)

    if not self.log_batch:
      return

    # log all batch computed metrics
    for stat_name, stat_classes in stats_dict.items():

      # if we are doing pq and it's a list (i.e. has rq and sq)
      if stat_name == 'PQ': # and isinstance( stat_classes, list ):
        self.log_metric( pl_module, f'{stage}_batch_allPQ', stat_classes['all']['pq'] )
        self.log_metric( pl_module, f'{stage}_batch_allRQ', stat_classes['all']['rq'] )
        self.log_metric( pl_module, f'{stage}_batch_allSQ', stat_classes['all']['sq'] )
        self.log_metric( pl_module, f'{stage}_batch_thingsPQ', stat_classes['things']['pq'] )
        self.log_metric( pl_module, f'{stage}_batch_thingsRQ', stat_classes['things']['rq'] )
        self.log_metric( pl_module, f'{stage}_batch_thingsSQ', stat_classes['things']['sq'] )
        self.log_metric( pl_module, f'{stage}_batch_stuffPQ', stat_classes['stuff']['pq'] )
        self.log_metric( pl_module, f'{stage}_batch_stuffRQ', stat_classes['stuff']['rq'] )
        self.log_metric( pl_module, f'{stage}_batch_stuffSQ', stat_classes['stuff']['sq'] )
        continue

      # log single class/averaged metrics
      if stat_classes.nelement() == 1:
        self.log_metric(pl_module, f'{stage}_batch_{stat_name}', stat_classes)
        continue

      # log multi class metrics
      if stat_name == 'conf_matrix':
        cm_img = tb_confusion_matrix_img(stat_classes, self.class_labels)
        if pl_module.logger is not None:
          pl_module.logger.experiment.add_image(f'{stage}_batch_{stat_name}',cm_img, batch_idx)
        continue

      # Weighed average stat
      avg_str = 'wavg'
      if torch.all(self.class_weights == 1):
        avg_str = 'avg'
      self.log_metric(pl_module, f'{stage}_batch_{stat_name}_{avg_str}', self.weightedAVG(stat_classes))

      # Class-wise stat
      for label, s in zip(self.class_labels, stat_classes):
        self.log_metric(pl_module, f'{stage}_batch_{stat_name}_{label}', s)

  def log_metrics_epoch(self, pl_module, stage=''):
    stats = self.compute()
    if not self.log_epoch:
      return
    for stat_name, stat in stats.items():

      # class related stats/scores
      if not isinstance(stat,list):
        # General stats
        self.log_metric(pl_module, f'{stage}_epoch_{stat_name}', stat)
        continue

      stat_avg, stat_classes = stat
      # log single class/averaged metrics
      if stat_classes.nelement() == 1:
        self.log_metric(pl_module, f'{stage}_epoch_{stat_name}', stat_classes)
        continue

      # log multi class metrics
      if stat_name == 'conf_matrix':
        cm_img = tb_confusion_matrix_img(stat_classes, self.class_labels)
        if pl_module.logger is not None:
          pl_module.logger.experiment.add_image(f'{stage}_epoch_{stat_name}',cm_img, pl_module.current_epoch)
        continue

      # Weighed average stat
      avg_str = 'wavg'
      if torch.all(self.class_weights == 1):
        avg_str = 'avg'
      self.log_metric(pl_module, f'{stage}_epoch_{stat_name}_{avg_str}', stat_avg)
      # Class-wise stat
      for label, stat_class in zip(self.class_labels, stat_classes):
        self.log_metric(pl_module, f'{stage}_epoch_{stat_name}_{label}', stat_class)

  def log_metric(self, pl_module, metric_name, metric):
    pl_module.log(metric_name, metric)

    if any([metric_name in pbm for pbm in self.pbar_metrics]):
      metric_name = '_'.join(metric_name.split('_')[-2:])
      pl_module.log(metric_name, metric, prog_bar=True)

    if self.log_best_metrics:
      criterion = [c for name,c in self.best_criterions.items() if name in metric_name]
      if not criterion:
        return
      # Initialize best metric if not already
      if metric_name not in self.best_metrics.keys():
        self.best_metrics[metric_name] = metric
      # Replace metric if better than pervious one
      else:
        self.best_metrics[metric_name] = criterion[0](self.best_metrics[metric_name], metric)
      # log best metric
      pl_module.log(f'{metric_name}_best', self.best_metrics[metric_name])

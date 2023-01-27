
import torch
import pytorch_lightning as pl

# Import the updated evaluator to use...
from modules.evaluator_base import EvaluatorTrainCallbacks
from modules.evaluator_base import EvaluatorValCallbacks
from modules.evaluator_base import EvaluatorTestCallbacks



class ThunderModule(pl.LightningModule):
  """
  The following is just a template, you can and should feel free to change it for your particular problem.

  However, for some tasks the only added complexity will be to change the "network" (self.net) in the 
  "template" and so just inheriting and defining your particular "network" (self.net).
  """

  #############################################
  def __init__(self, cfg={}):
    super(ThunderModule, self).__init__()
    """
    Store the configuration, define network, evaluator, and loss criterion placeholders
    """
    self.cfg               = cfg
    # you need to override these for your module file for your specific architecture
    self.net               = None 
    self.evaluator         = None
    self.criterion         = None

  #############################################
  ### Infrastructure Config
  #############################################
  def configure_callbacks(self):
    # Create independant evaluators for train, test & val to allow independet
    # stage-wise metrics bookkeeping
    assert(self.evaluator is not None)

    self.train_evaluator = self.evaluator(cfg=self.cfg)
    self.val_evaluator = self.evaluator(cfg=self.cfg)
    self.test_evaluator = self.evaluator(cfg=self.cfg)
    # Append evaluator callbacks for all stages
    return [EvaluatorTrainCallbacks(self.train_evaluator),
            EvaluatorValCallbacks(self.val_evaluator),
            EvaluatorTestCallbacks(self.test_evaluator)]



  #############################################
  # Forward function for the network
  def forward(self, x):
    """
    Pass the data (x) through the network.
    """
    return self.net(x)


  #############################################
  # Define the otimizer to use
  def configure_optimizers(self):
    """
    Do something to configure the optimizer and return it..
    """

    # Grab the stuff from the config file
    self.step_size     = self.cfg['optimizer']['step_size']
    self.gamma         = self.cfg['optimizer']['gamma']
    self.base_lr       = self.cfg['optimizer']['base_lr']

    # Replace this code at some point
    self.optimizer      = torch.optim.Adam(self.parameters(), lr=self.base_lr)
    self.lr_scheduler   = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                          step_size=self.step_size,
                                                          gamma=self.gamma)

    return {'optimizer': self.optimizer, 'lr_scheduler': self.lr_scheduler}



  #############################################
  # The training step and training step end. This allows
  # more flexible use of multiple GPUs (the end step takes
  # the previously separated batches and sticks them back 
  # together).
  def training_step(self, batch, batch_idx):
    """
    Do the training step, usually grab the labels, push the data through the network
    and then calculate the loss. Return the loss at the end.
    """
    assert(self.net is not None)
    assert(self.criterion is not None)
    # Get the labels, predictions and produce a loss
    labels                 = batch['gt']
    preds                  = self(batch['rgb'])
    loss                   = self.criterion(preds['out'], labels)

    return {'loss':loss, 'preds':preds, 'labels':labels}    


  #############################################
  # Validation step
  def validation_step(self, batch, batch_idx):
    """
    Do the validation step, usually grab the labels, push the data through the network
    and then calculate the loss. Return the loss at the end.
    """
    assert(self.net is not None)

    labels      = batch['gt']
    preds       = self(batch['rgb'])
    loss        = self.criterion(preds['out'], labels)

    return {'loss':loss, 'preds':preds, 'labels':labels}


  #############################################
  # Evaluation step
  def test_step(self, batch, batch_idx):
    assert(self.net is not None)

    labels      = batch['gt']
    preds       = self(batch['rgb'])

    return {'preds':preds, 'labels':labels}




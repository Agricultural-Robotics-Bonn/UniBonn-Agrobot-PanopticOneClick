
import pytorch_lightning.metrics.functional as FM
import torch

from ..thunder_module import ThunderModule
from .model import UNet_tag_point
from modules.evaluator_base import EvaluatorValCallbacks, EvaluatorTestCallbacks
from modules.panoptic_evaluator import PanopticEvaluator
from modules.panoptic_weakseg_evaluator import PanopticWeaksegEvaluator

from models.utils_panoptic.post_processing import get_panoptic_segmentation



#####
class PLModule(ThunderModule):
  """
  The work horse that interfaces to PyLightning.
  """

  def __init__(self, cfg={}, evaluator=None):
    """
    There's an assumed structure where we have
      - tag_segmentation, tag_regression, and semantic_segmentation

      - tag_segmentation: this is the tag point segmentation (small round circles)
      - tag_regression: is the distance to the tag point
      - semantic_segmentation: is a regular semantic segmentation head
    """
    super(PLModule, self).__init__(cfg=cfg)
    self.cfg = cfg
    if 'nocnt' in self.cfg['network']['submodel']:
      self.head_keys = ['reg', 'sem']
    else:
      self.head_keys = ['cnt', 'reg', 'sem']
    self.head_filters_set = []
    self.head_types = []
    self.output_channel_set = []
    self.head_dict = {}
    self.loss_dict = torch.nn.ModuleDict()

    # Set up the network
    print(cfg['network'])
    self.select_network(cfg)
    
  #############################################
  ### Assemble the whole batch if it is a dict of lists
  #############################################
  def _assemble_batch( self, batch ):
    """ 
      Assemble a batch if it is a dict of lists 
      need this to stack together click positions of different lenghts
    """
    output = {}
    for k in batch.keys():
      if k == 'RGB_fname':
        output[k] = batch[k]
        continue
      if k == 'keypoints':
        continue
      if k == 'cntcoords':
        output[k] = batch[k]
      else:
        output[k] = torch.stack( batch[k], dim=0 )
    output['panoptic'] = output['gt']
    return output


  def _calculate_loss( self, preds, batch ):
    loss = 0
    for c_head, c_entry in self.head_dict.items():
      c_preds = preds[c_head]
      c_labels = self.select_labels(c_head, batch)
      if c_entry['loss_name'] == 'l1_valid_regression':
          idx = torch.nonzero(c_labels, as_tuple=True)  # valid pixels only
          target_offsets = c_labels[idx]
          pred_offsets = c_preds[idx]
          c_loss = self.loss_dict[c_head]( target_offsets, pred_offsets )
      else:
        c_loss = self.loss_dict[c_head](c_preds, c_labels) 
      # Accumulate
      loss = loss + c_loss*c_entry['loss_weight']

    return loss


  #############################################
  ### Process the panoptic metric tensors
  #############################################
  def process_panoptic( self, preds, batch ):
    # preamble
    if 'post_processing' in self.cfg.keys():
      threshold = self.cfg['post_processing']['threshold'] if 'threshold' in self.cfg['post_processing'].keys() else 0.1
      nms_kernel = self.cfg['post_processing']['nms_kernel'] if 'nms_kernel' in self.cfg['post_processing'].keys() else 7
      top_k = self.cfg['post_processing']['top_k'] if 'top_k' in self.cfg['post_processing'].keys() else 200
      label_divisor = self.cfg['post_processing']['top_k'] if 'top_k' in self.cfg['post_processing'].keys() else 1
      stuff_area = self.cfg['post_processing']['stuff_area'] if 'stuff_area' in self.cfg['post_processing'].keys() else 20
      void_label = self.cfg['post_processing']['void_label'] if 'void_label' in self.cfg['post_processing'].keys() else 0
    else: # these values are directly from the paper.
      threshold = 0.1
      nms_kernel = 7
      top_k = 200
      label_divisor = 1
      stuff_area = 20
      void_label = 0
    # get the other stuff required for the panoptic segmentation requirements
    thing_ids = list( range( 1, self.cfg['dataset']['class_num'] ) )
    # iterate over the batch as it is done by single elements
    panoptic_preds = None
    for s, c, o in zip( preds['sem'], preds['cnt'], preds['reg'] ):
      s = torch.argmax( s, dim=0 ).unsqueeze( 0 )
      _, _, _, imap = get_panoptic_segmentation( s, c, o,
                                                    thing_ids,
                                                    label_divisor,
                                                    stuff_area,
                                                    void_label,
                                                    threshold=threshold,
                                                    nms_kernel=nms_kernel,
                                                    top_k=top_k,
                                                    foreground_mask=None )
      # create teh smap and store it.
      smap = torch.zeros_like( imap )
      smap[imap>0] = 1
      ppreds = torch.cat( (smap, imap), axis=0 ).unsqueeze( 0 )
      # store the outputs
      try:
        panoptic_preds = torch.cat( (panoptic_preds, ppreds) )
      except:
        panoptic_preds = ppreds
    return panoptic_preds


  #############################################
  ### Process the panoptic metric tensors but when no cnt head is available (differnet output)
  #############################################
  def process_panoptic_weak( self, preds, batch ):
    '''
      basically, this function is used when no cnt head is available.
      original one above is also used for some other cases of weakseg, e.g. when input clicks are missing, hence cnt heatmap is predicted
    '''
    # preamble
    if 'post_processing' in self.cfg.keys():
      threshold = self.cfg['post_processing']['threshold'] if 'threshold' in self.cfg['post_processing'].keys() else 0.1
      nms_kernel = self.cfg['post_processing']['nms_kernel'] if 'nms_kernel' in self.cfg['post_processing'].keys() else 7
      top_k = self.cfg['post_processing']['top_k'] if 'top_k' in self.cfg['post_processing'].keys() else 200
      label_divisor = self.cfg['post_processing']['top_k'] if 'top_k' in self.cfg['post_processing'].keys() else 1
      stuff_area = self.cfg['post_processing']['stuff_area'] if 'stuff_area' in self.cfg['post_processing'].keys() else 20
      void_label = self.cfg['post_processing']['void_label'] if 'void_label' in self.cfg['post_processing'].keys() else 0
    else: # these values are directly from the paper.
      threshold = 0.1
      nms_kernel = 7
      top_k = 200
      label_divisor = 1
      stuff_area = 20
      void_label = 0
    # get the other stuff required for the panoptic segmentation requirements
    thing_ids = list( range( 1, self.cfg['dataset']['class_num'] ) )
    # iterate over the batch as it is done by single elements
    ############
    '''
      passing the centercoords instead of the predicted heatmap here as we do not have this head
      adapted post_processing.py to treat them as if they were extracted from heatmaps in there...
    '''
    preds['cnt'] = batch['cntcoords']
    threshold = 0
    ############
    for s, c, o in zip( preds['sem'], preds['cnt'], preds['reg'] ):
      s = torch.argmax( s, dim=0 ).unsqueeze( 0 )
      _, center, _, imap = get_panoptic_segmentation( s, c, o,
                                                    thing_ids,
                                                    label_divisor,
                                                    stuff_area,
                                                    void_label,
                                                    threshold=threshold,
                                                    nms_kernel=nms_kernel,
                                                    top_k=top_k,
                                                    foreground_mask=None )

      try:
        cnt_preds = torch.cat( (cnt_preds, center), dim=0 )
        instance_preds = torch.cat( (instance_preds, imap), dim=0 )
      except:
        cnt_preds = center
        instance_preds = imap
    return cnt_preds, instance_preds




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
    # Get the labels, predictions and produce a loss
    if isinstance( batch['rgb'], list ):
      batch = self._assemble_batch( batch )
    preds = self(batch['rgb'])
    loss = self._calculate_loss( preds, batch )
    self.log_dict( {'train_loss': loss}, on_step=False, on_epoch=True, sync_dist=True )

    return {'loss':loss, 'preds':preds, 'labels':batch}    


  #############################################
  # Validation step
  def validation_step(self, batch, batch_idx):
    """
    Do the validation step, usually grab the labels, push the data through the network
    and then calculate the loss. Return the loss at the end.
    """
    if isinstance( batch['rgb'], list ):
      batch = self._assemble_batch( batch )
    preds = self(batch['rgb'])
    loss = self._calculate_loss( preds, batch )

    # calculate the static statistic
    p = torch.argmax( preds['sem'], dim=1 )
    m = FM.accuracy( p, batch['gt'][:,0,:,:] )
    self.log_dict( {'val_acc_epoch':m}, on_step=False, on_epoch=True, sync_dist=True )

    ## get the appropriate keys for the post-processing for labels and predictions
    if 'nocnt' in self.cfg['network']['submodel']:
      cnt_preds, panoptic_preds = self.process_panoptic_weak( preds, batch )  # this is essentially not returning panoptic preds but instance preds based on panoptic
      preds['panoptic'] = panoptic_preds
      preds['cntcoords'] = cnt_preds
    else:
      preds['panoptic'] = self.process_panoptic( preds, batch )

    preds['out'] = preds['sem']
    return {'epoch':self.current_epoch, 'loss':loss, 'preds':preds, 'labels':batch}


  #############################################
  # Evaluation step
  def test_step(self, batch, batch_idx):
    if isinstance( batch['rgb'], list ):
      batch = self._assemble_batch( batch )
    preds = self(batch['rgb'])
    loss = self._calculate_loss( preds, batch )

    # calculate the static statistic
    p = torch.argmax( preds['sem'], dim=1 )
    m = FM.accuracy( p, batch['gt'][:,0,:,:] )
    self.log_dict( {'val_acc_epoch':m}, on_step=False, on_epoch=True, sync_dist=True )

    ## get the appropriate keys for the post-processing for labels and predictions
    if 'nocnt' in self.cfg['network']['submodel']:
      cnt_preds, panoptic_preds = self.process_panoptic_weak( preds, batch )  # this is essentially not returning panoptic preds but instance preds based on panoptic
      preds['panoptic'] = panoptic_preds
      preds['cntcoords'] = cnt_preds
    else:
      panoptic_preds = self.process_panoptic( preds, batch )
      preds['panoptic'] = panoptic_preds

    # return everything
    preds['out'] = preds['sem']
    return {'loss':loss, 'preds':preds, 'labels':batch}
      



  #############################################
  ### Infrastructure Config
  #############################################
  def configure_callbacks(self):
    # Create independant evaluators for train, test & val to allow independet
    # stage-wise metrics bookkeeping

    self.callbacks = []
    self.evaluator_modules = torch.nn.ModuleList()

    ### Select the evaluator based on the type of the head that we have...
    for c_head in self.head_keys:
      c_entry = self.cfg['network']['head_dictionary'][c_head] # Shortcut into the dictionary for this head
      if c_entry['type'] == 'segmentation':

        # Validation evaluator
        val_evaluator = PanopticWeaksegEvaluator(self.cfg) if 'nocnt' in self.cfg['network']['submodel'] else PanopticEvaluator(self.cfg)
        self.evaluator_modules.append(val_evaluator)
        self.callbacks.append(EvaluatorValCallbacks(val_evaluator))

        # Evaluation evaluator
        eval_evaluator = PanopticWeaksegEvaluator(self.cfg) if 'nocnt' in self.cfg['network']['submodel'] else PanopticEvaluator(self.cfg)
        self.evaluator_modules.append(eval_evaluator)
        self.callbacks.append(EvaluatorTestCallbacks(eval_evaluator))

    return self.callbacks




  #########################################################
  ##### HELPER FUNCTIONS: mostly initialisation stuff #####
  #########################################################

  #
  def select_network(self, cfg):
    """
    For each of the heads, grab the class weights and the loss type to give 
    this to the training schemes.

    Then, do a switch statement to choose between the network variants.
    """


    # Be sure the config has the required keys (self.head_keys)!!
    for c_head in self.head_keys: 
      c_entry = cfg['network']['head_dictionary'][c_head] # Shortcut into the dictionary for this head
      self.head_dict[c_head] = {'loss_name': c_entry['loss_name'],
                                'filters': c_entry['filters'],
                                'output_channels': c_entry['output_channels'],
                                'type': c_entry['type']}

      # Grab the class weights if they exist and ensure they're the right size. 
      # Make the relevant loss at the same time.
      try:
        class_weights = c_entry['class_weights']
        print("For", c_head, "class weights are:", class_weights)
        self.head_dict[c_head]['class_weights'] = class_weights
        loss = self.select_loss(self.head_dict[c_head]['loss_name'], class_weights=class_weights)
        self.head_dict[c_head]['class_ids'] = c_entry['class_ids']
        self.head_dict[c_head]['class_labels'] = c_entry['class_labels']

      except KeyError: # Otherwise there are no class weights...
        loss = self.select_loss(self.head_dict[c_head]['loss_name'])

      self.loss_dict[c_head] = loss
      self.head_dict[c_head]['loss_weight'] = c_entry['loss_weight']

      # Put the information together in a list to pass over to the initialisation of the network
      self.head_filters_set.append(self.head_dict[c_head]['filters'])
      self.output_channel_set.append(self.head_dict[c_head]['output_channels'])
      self.head_types.append(self.head_dict[c_head]['type'])

    # Set up the network with multiple heads
    if any( cfg['network']['submodel']==x for x in ( 'UNet', 'UNet_nocnt' ) ):
      self.net = UNet_tag_point(output_channel_set=self.output_channel_set,
                                head_filter_set=self.head_filters_set,
                                head_types=self.head_types,
                                head_keys=self.head_keys,
                                **(cfg['network']) )


  def select_loss(self, loss_name, class_weights=None, focal_value=3):
    """
    Go through the kind of losses passed to us and set up the loss as requested.
    """
    if loss_name == "class_weighted_xentropy":
      class_weights = torch.tensor(class_weights)
      loss = torch.nn.CrossEntropyLoss(weight=class_weights)
    elif loss_name == "l1_valid_regression":
      loss = torch.nn.L1Loss() # Regression loss
    elif loss_name == "l2":
      loss = torch.nn.MSELoss()
    else:
      print("PLModule::select_loss invalid loss:", loss_name)
      print("PLModule::select_loss invalid loss:", loss_name)
      exit()
    return loss


  def select_labels(self, head_name, labels):
    """
    Go through the data from the data loader (in batch) and return what we 
    want for the given head_name.
    """
    if head_name == "cnt":
      labels = labels['cnt']
    elif head_name == "reg":
      labels = labels['reg'] # This is a tuple of x,y so put them together
    elif head_name == "sem":
      labels = labels['gt'][:,0,:,:]
    else:
      print("PLModule::select_loss invalid head_name:", head_name)
      exit()
    return labels


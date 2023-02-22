"""
  A data loader based on the coco module to load SB20 and CN20 datasets.
"""
import os
import sys

import numpy
from skimage.io import imread

import yaml

from pycocotools.coco import COCO

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tvtransforms
import pytorch_lightning as pl

import dataloaders.modules.numpyTransforms as Mytrans
import dataloaders.modules.utils as Myutils


# Disable
def blockPrint():
    sys.stdout = open( os.devnull, 'w' )
# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


class CoCo( Dataset ):

  def __init__(self,
               root,
               subset,
               dataset_name,
               transforms=None,
               class_type='plant',
               extension='png',
               skip_LUT=False,
               return_keypoints=False,
               output_size=None,
               ):

    self._root_dir = root
    self._extension = extension
    self.dataset_name = dataset_name
    self.subset = subset
    self.transforms = transforms
    self.return_keypoints = return_keypoints
    self.output_size = output_size

    assert self.subset == 'train' or self.subset == 'valid' or self.subset == 'eval'

    # Directories and root addresses
    self.annotation_files_dir = os.path.join( self._root_dir, '' if 'staticdata' in self._root_dir else 'datasets', self.dataset_name, self.dataset_name + '.json' )
    self.dataset_config_list_dir = os.path.join( self._root_dir, '' if 'staticdata' in self._root_dir else 'datasets', self.dataset_name, self.dataset_name + '.yaml' )
    with open(self.dataset_config_list_dir) as stream:
        self.dataset_config = yaml.safe_load( stream )
    self.image_sets =  self.dataset_config["image_sets"]

    blockPrint()
    # Initialize COCO api for instance annotations
    self.coco = COCO( self.annotation_files_dir )
    enablePrint()

    # create look up table for the classes or sub-classes
    if not skip_LUT:
      self.LUT = dict()
      for id, c in self.coco.cats.items():
        if class_type == 'plant':
          self.LUT[id] = 1
        elif class_type == 'super': # crop, weed, unknown
          if c['supercategory'] == 'crop':
            self.LUT[id] = 1
          elif c['supercategory'] == 'weed':
            self.LUT[id] = 2
          else:
            self.LUT[id] = 3
        elif class_type == 'subassuper': # SB, Ch, Th, Bi, PE, unknown, Chy, An
          l = ['SB', 'Ch', 'Th', 'Bi', 'Pe', 'unknown', 'Chy', 'An']
          self.LUT[id] = l.index( c['name'] )+1
        else:
          assert False, 'The class type is incorrect, only plant super or subassuper exists'

  def getTargetID(self, index):
    img_set_ids = self.image_sets[self.subset][index]
    img_list_idx = next( (index for (index, d) in enumerate( self.coco.dataset["images"] ) if d["id"] == img_set_ids), None )
    image_id = self.coco.dataset["images"][img_list_idx]["id"]
    return image_id

  def getImgGT( self, index ):
    # get meta data of called image (ID = index)
    self.img_id = self.getTargetID( index )
    self.img_metadata = self.coco.loadImgs( self.img_id )[0]
    path = self.img_metadata['path'][1:]
    path_abs = os.path.join( self._root_dir, path )
    if '_subset' in self.dataset_name and '_subset' not in path_abs:
      path_abs = path_abs.replace( 'CKA_sugar_beet_2020', 'CKA_sugar_beet_2020_subset' )
      path_abs = path_abs.replace( 'CKA_corn_2020', 'CKA_corn_2020_subset' )
    img = imread( path_abs )/255.
    # load the masks and labels
    cat_ids = self.coco.getCatIds()
    anns_ids = self.coco.getAnnIds( imgIds=self.img_id, catIds=cat_ids, iscrowd=None )
    # get the annotations for the current image
    anns = self.coco.loadAnns(anns_ids)
    # get keypoints, the semantic mask and the imap
    keypoints = []
    itr = 1
    semmask = numpy.zeros( (self.img_metadata['height'], self.img_metadata['width']) ).astype( numpy.uint8 )
    imap = numpy.zeros( (self.img_metadata['height'], self.img_metadata['width']) ).astype( numpy.uint8 )
    for ann in anns:
      if not ann['segmentation']:
        continue
      if self.return_keypoints:
        keypoints_ins = ann["keypoints"] if "keypoints" in ann.keys() else None
        if keypoints_ins is not None:
          keypoints_ins = [ keypoints_ins[i:i+3] for i in range(0, len(keypoints_ins), 3) ]
          bbox = ann["bbox"]
          keypoints_ins = [ kp for kp in keypoints_ins if not ( kp[0]<bbox[0] or kp[0]>(bbox[0]+bbox[2]) or kp[1]<bbox[1] or kp[1]>(bbox[1]+bbox[3]) ) ]
        keypoint = keypoints_ins[0] if ( keypoints_ins is not None and len(keypoints_ins) > 0 ) else None
        keypoints.append( keypoint )
      ann_mask = self.coco.annToMask( ann )
      semmask[ann_mask>0] = self.LUT[ann['category_id']]
      imap[ann_mask>0] = itr
      itr += 1
    panoptic = numpy.dstack( (semmask, imap) )
    return img, panoptic, path, keypoints
  
  
  @staticmethod
  def resizeKeypoints( keypoints, input_size, output_size ):
    # fetching original data (from last images loaded metadata)
    w_org = input_size[1]
    h_org = input_size[0]
    # fetching resize props
    w_new = output_size[1]
    h_new = output_size[0]
    # ratios
    w_rat = w_org / w_new
    h_rat = h_org / h_new
    # transform
    keypoints_trans = []
    for keypoint in keypoints:
      if keypoint is not None:
        x_new = round( keypoint[0] / w_rat )
        y_new = round( keypoint[1] / h_rat )
        keypoints_trans.append( [ x_new, y_new, keypoint[2] ]  )  # staying with original format here
      else:
        keypoints_trans.append( None )
    return keypoints_trans


  def __getitem__( self, index ):
    # get the image, panoptic, and path
    img, panoptic, img_path, keypoints = self.getImgGT( index )
    # transform where required.
    sample = { 'rgb':img.copy(), 'gt':panoptic.copy() }
    if self.return_keypoints:
      input_size = ( self.__dict__['img_metadata']['height'], self.__dict__['img_metadata']['width'] )
      keypoints = self.resizeKeypoints( keypoints, input_size, self.output_size )
      sample['keypoints'] = keypoints
    if self.transforms is not None:
      sample = self.transforms( sample )
    sample['RGB_fname'] = os.path.basename( img_path )
    return sample


  def __len__( self ):
    return len( self.image_sets[self.subset] )


class Parser( pl.LightningDataModule ):
  def __init__( self, config ):
    super( Parser, self ).__init__()
    # config file
    self.config = config
    if config['dataset']['transforms']['use']:
      train_ts, infer_ts = list(), list()
      for k, v in config['dataset']['transforms'].items():
        # MeanStdNorm
        if k == 'meanstdnorm' and v['use']:
          train_ts.append( Mytrans.MeanStdNorm( RGB_mean_arr=config['dataset']['transforms']['meanstdnorm']['RGB_mean_arr'],
                                                RGB_std_arr=config['dataset']['transforms']['meanstdnorm']['RGB_std_arr'] ) )
          infer_ts.append( Mytrans.MeanStdNorm( RGB_mean_arr=config['dataset']['transforms']['meanstdnorm']['RGB_mean_arr'],
                                                RGB_std_arr=config['dataset']['transforms']['meanstdnorm']['RGB_std_arr'] ) )
        # Rescale
        if k == 'rescale' and v['use']:
          train_ts.append( Mytrans.Rescale( output_size=config['dataset']['transforms']['rescale']['output_size'], ispanoptic=config['dataset']['transforms']['rescale']['ispanoptic'] ) )
          infer_ts.append( Mytrans.Rescale( output_size=config['dataset']['transforms']['rescale']['output_size'], ispanoptic=config['dataset']['transforms']['rescale']['ispanoptic'] ) )
        # Panoptic
        if k == 'panoptic' and v['use']:
          train_ts.append( Mytrans.Panoptic( radius=config['dataset']['transforms']['panoptic']['radius'],
                                             blur=config['dataset']['transforms']['panoptic']['blur'],
                                             noiserange=config['dataset']['transforms']['panoptic']['noiserange'],
                                             missing_click_perc=config['dataset']['transforms']['panoptic']['missing_click_perc'] if 'missing_click_perc' in config['dataset']['transforms']['panoptic'].keys() else 0,
                                            ) )
          infer_ts.append( Mytrans.Panoptic( radius=config['dataset']['transforms']['panoptic']['radius'],
                                             blur=config['dataset']['transforms']['panoptic']['blur'],
                                             noiserange=None,
                                             missing_click_perc=config['dataset']['transforms']['panoptic']['missing_click_perc'] if 'missing_click_perc' in config['dataset']['transforms']['panoptic'].keys() else 0,
                                             randomize_missing=False,
                                             ) )
        # AppendCentersAsInput
        if k == 'cntasinput' and config['dataset']['transforms']['cntasinput']['use']:
          train_ts.append( Mytrans.AppendCentersAsInput( ) )
          infer_ts.append( Mytrans.AppendCentersAsInput( ) )
        # ToTensor (MUST be at the end)
        if k == 'totensor' and v['use']:
          train_ts.append( Mytrans.ToTensor() )
          infer_ts.append( Mytrans.ToTensor() )
      if len( train_ts ) > 0:
        print(f"Train transforms: {[type(tts).__name__ for tts in train_ts]}")
        train_transforms = tvtransforms.Compose( train_ts )
      if len( infer_ts ) > 0:
        print(f"Infer transforms: {[type(its).__name__ for its in infer_ts]}")
        infer_transforms = tvtransforms.Compose( infer_ts )

    # other parameters for the dataloader
    extension = config['dataset']['extension'] if 'extension' in config['dataset'] else 'png'
    class_type = config['dataset']['class_type'] if 'class_type' in config['dataset'] else 'plant'
    return_keypoints = config['dataset']['return_keypoints'] if 'return_keypoints' in config['dataset'] else False
    output_size = config['dataset']['transforms']['rescale']['output_size']
    # create the dataloader
    self.loadertrain, self.loadervalid, self.loadereval = None, None, None
    if 'train' in config['dataset']['subsets']:
      self.loadertrain = CoCo( config['dataset']['location'], 'train', config['dataset']['coconame'],
                               transforms=train_transforms,
                               extension=extension,
                               class_type=class_type,
                               return_keypoints=return_keypoints,
                               output_size=output_size,
                               )
      print( '******* Training Samples =', len( self.loadertrain ), '*************' )

    if 'valid' in config['dataset']['subsets']:
      self.loadervalid = CoCo( config['dataset']['location'], 'valid', config['dataset']['coconame'],
                               transforms=infer_transforms,
                               extension=extension,
                               class_type=class_type,
                               return_keypoints=return_keypoints,
                               output_size=output_size,
                               )
      print( '******* Validation Samples =', len( self.loadervalid ), '*************' )

    if 'eval' in config['dataset']['subsets']:
      self.loadereval = CoCo( config['dataset']['location'], 'eval', config['dataset']['coconame'],
                              transforms=infer_transforms,
                              extension=extension,
                              class_type=class_type,
                              return_keypoints=return_keypoints,
                              output_size=output_size,
                              )
      print( '******* Evaluation Samples =', len( self.loadereval ), '*************' )

  # get the training parser
  def train_dataloader( self ):
    if self.loadertrain is None:
      return self.loadertrain
    return DataLoader( self.loadertrain,
                        batch_size=self.config['dataloader']['batch_size'],
                        shuffle=self.config['dataloader']['shuffle'],
                        num_workers=self.config['dataloader']['workers_num'],
                        drop_last=self.config['dataloader']['drop_last'],
                        collate_fn=Myutils.collate_variable_masks, )

  # get the validation parser
  def val_dataloader( self ):
    if self.loadervalid is None:
      return self.loadervalid
    return DataLoader( self.loadervalid,
                        batch_size=self.config['dataloader']['batch_size'],
                        shuffle=False,
                        num_workers=self.config['dataloader']['workers_num'],
                        drop_last=False,
                        collate_fn=Myutils.collate_variable_masks, )

  # get the evaluation parser
  def test_dataloader( self ):
    if self.loadereval is None:
      return self.loadereval
    return DataLoader( self.loadereval,
                        batch_size=self.config['dataloader']['batch_size'],
                        shuffle=False,
                        num_workers=self.config['dataloader']['workers_num'],
                        drop_last=False,
                        collate_fn=Myutils.collate_variable_masks, )

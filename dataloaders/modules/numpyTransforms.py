"""
Some transform examples for this data.
"""
import numpy
import skimage.transform
from skimage import filters
from skimage.draw import disk
import torch
import time
import random
import copy

from .weakseg_utils import WeakSegUtils



class MeanStdNorm(object):
  """Normalise the image by the mean and standard deviation

  Args:
    RGB_mean_arr: the mean of the images to be normalised to.
    RGB_std_arr:  the std dev of the images to be normalised to.
  """

  def __init__(self, RGB_mean_arr=numpy.array([0.5, 0.5, 0.5]),
                     RGB_std_arr=numpy.array([0.5, 0.5, 0.5]),
              ):
    self.RGB_mean_arr = RGB_mean_arr
    self.RGB_std_arr = RGB_std_arr

  def __call__(self, sample):
    image, GT = sample['rgb'], sample['gt']

    image = (image - self.RGB_mean_arr)/self.RGB_std_arr

    ret_dict = dict(
                rgb=image,
                gt=GT,
                )
    if 'boxes' in sample.keys():
      ret_dict['boxes'] = sample['boxes']
    for k in sample.keys():
      if k not in ['rgb', 'gt', 'boxes']:
        ret_dict[k] = sample[k]
    return ret_dict



class Rescale(object):
  """Rescale the image in a sample to a given size.

  Args:
    output_size (tuple or int): Desired output size. If tuple, output is
        matched to output_size. If int, smaller of image edges is matched
        to output_size keeping aspect ratio the same.
  """

  def __init__( self, output_size, min_pixels=20, ispanoptic=False ):
    assert isinstance(output_size, (int, tuple, list))
    self.output_size = output_size
    self.min_pixels = min_pixels
    self.ispanoptic = ispanoptic

  def __call__( self, sample ):
    image, GT = sample['rgb'], sample['gt']
    sub_labels = sample['sub_labels'] if 'sub_labels' in sample.keys() else None
    h, w = image.shape[:2]
    if isinstance(self.output_size, int):
      if h > w:
        new_h, new_w = self.output_size * h / w, self.output_size
      else:
        new_h, new_w = self.output_size, self.output_size * w / h
    else:
      new_h, new_w = self.output_size

    new_h, new_w = int(new_h), int(new_w)
    img = skimage.transform.resize( image, (new_h, new_w) )
    if GT is not None:
      GT = skimage.transform.resize( GT, (new_h, new_w), anti_aliasing=False,
                               anti_aliasing_sigma=None,
                               order = 0, preserve_range=True )

    # now remove gt's that are no longer in the image.
    if self.ispanoptic or len( GT.shape )==2:
      omask = GT
    else:
      sls, omask = None, None
      for g in range( GT.shape[2] ):
        (x,y) = numpy.where( GT[:,:,g]>0 )
        ux, uy = numpy.unique( x ), numpy.unique( y )
        if ux.shape[0] > 1 and uy.shape[0] > 1 and x.shape[0] > self.min_pixels:
          try:
            omask = numpy.dstack( (omask, GT[:,:,g]) )
            if sub_labels is not None:
              sls.append( sub_labels[g] )
          except:
            omask = GT[:,:,g]
            if sub_labels is not None:
              sls = [sub_labels[g]]

    if omask is None:
      omask = numpy.zeros( (img.shape[0], img.shape[1], 1) )
    ret_dict = dict(
                rgb=img, gt=omask
                )
    if sub_labels is not None:
      ret_dict['sub_labels'] = sls

    if 'boxes' in sample.keys():
      boxes = []
      for i in range( omask.shape[2] ):
        pos = numpy.where( omask[:,:,i] )
        xmin = numpy.min( pos[1] )
        xmax = numpy.max( pos[1] )
        ymin = numpy.min( pos[0] )
        ymax = numpy.max( pos[0] )
        boxes.append( [xmin, ymin, xmax, ymax] )
      boxes = numpy.array( boxes )
      ret_dict['boxes'] = boxes
    for k in sample.keys():
      if k not in ['rgb', 'gt', 'boxes', 'sub_labels']:
        ret_dict[k] = sample[k]
    return ret_dict



class Panoptic( object ):
  """
    Creates the center and regression from the panoptic imap

    !!! Note, do this after reshape crop etc.
  """

  def __init__( self, radius=1, blur=[9,9], noiserange=None, missing_click_perc=0, randomize_missing=True, ):
    self.radius = radius
    self.blur = blur
    self.noiserange = noiserange
    self.missing_click_perc = missing_click_perc
    self.randomize_missing = randomize_missing
    
  def __call__( self, sample ):
    if 'keypoints' in sample.keys():
      weaksegutils = WeakSegUtils()
    # get the ground truth imap
    imap = sample['gt'][:,:,1]
    # storage
    first = True
    tags = numpy.zeros( (imap.shape[0], imap.shape[1]) )
    reg = numpy.zeros( (imap.shape[0], imap.shape[1], 2) )
    cntcoords = []
    cntcoords_selection = []
    cntcoords_missing = []
    n = len(numpy.unique(imap)) -1
    if not self.randomize_missing:
      ## disable randomness by using a fixed seed (2476268, that's "agrobot" on a phone dial)
      random.seed( 2476268 )
    missing = random.sample( range( 0, n ), int(round( n*self.missing_click_perc )) )
    ## return to randomness (using system time in ns)
    random.seed( int(time.time()*10000000) )
    # create the centers
    for i, u in enumerate(numpy.unique( imap )):
      if u == 0:
        continue
      i = i-1  # to make up for skipped u==0
      rr, cc = numpy.where( imap==u )
      if 'keypoints' in sample.keys():  # basically if doing weak stuff (should always have keypoints in this case)
        keypoint = sample['keypoints'][i]
        click = weaksegutils.createClick( gt=numpy.where( imap==u,1,0 ),
                                          keypoint=keypoint,
                                          noiserange=self.noiserange )
        r, c = click[0], click[1]
      else:
        r, c = int( rr.mean() ), int( cc.mean() )
      cntcoords.append( (r,c) )
      cntcoords_selection.append( (r,c) if i not in missing else None )
      cntcoords_missing.append( (r,c) if i in missing else None )
      # create the radius
      r, c = disk( (r,c), self.radius, shape=imap.shape )
      # draw
      tag = numpy.zeros( (imap.shape[0], imap.shape[1]) )
      tag_selection = numpy.zeros( (imap.shape[0], imap.shape[1]) )
      tag_missing = numpy.zeros( (imap.shape[0], imap.shape[1]) )
      tag[r,c] = 1
      tag_selection[r,c] = 1 if i not in missing else 0
      tag_missing[r,c] = 1 if i in missing else 0
      # are we blurring?
      if self.blur:
        tag = filters.gaussian( tag, self.blur, preserve_range=True )
        tag_selection = filters.gaussian( tag_selection, self.blur, preserve_range=True )
        tag_missing = filters.gaussian( tag_missing, self.blur, preserve_range=True )
        tag /= tag.max()
        if i not in missing:
          tag_selection /= tag_selection.max()
        if i in missing:
          tag_missing /= tag_missing.max()
      # store the tags channelwise
      if first:
        tags = tag
        tags_selection = tag_selection
        tags_missing = tag_missing
        first = False
      else:
        tags = numpy.dstack( (tags, tag) )
        tags_selection = numpy.dstack( (tags_selection, tag_selection) )
        tags_missing = numpy.dstack( (tags_missing, tag_missing) )
      # create the regression
      cr = (rr.max() - rr.min())//2 + rr.min()
      cn = (cc.max() - cc.min())//2 + cc.min()
      # install the values into the tregression
      reg[rr,cc,0] = (cr-rr)
      reg[rr,cc,1] = (cn-cc)
    # get the centers as a single channel
    if len( tags.shape ) > 2:
      tags = numpy.clip( numpy.amax( tags, axis=2 ), 0., 1. )
      tags_selection = numpy.clip( numpy.amax( tags_selection, axis=2 ), 0., 1. )
      tags_missing = numpy.clip( numpy.amax( tags_missing, axis=2 ), 0., 1. )
    # store the values
    sample['cnt'] = tags
    sample['cnt_selection'] = tags_selection
    sample['cnt_missing'] = tags_missing
    sample['reg'] = reg
    sample['cntcoords'] = cntcoords
    return sample
    
    
    
class AppendCentersAsInput( object ):
  """
    Appends the cnt map to the input RGB

    !!!NOTE: Has to be used after Tag or Panoptic!!!!!!
  """

  def __init__( self, ):
    pass

  def __call__( self, sample ):
    rgb = sample['rgb']
    tag = copy.deepcopy( sample['cnt_selection'] )
    tag = numpy.expand_dims( tag, axis=2 )
    rgb_new = numpy.concatenate( ( rgb, tag ), axis=2 )
    sample['rgb'] = rgb_new
    return sample



class ToTensor(object):
  """Convert ndarrays in sample to Tensors."""

  def __call__(self, sample):
    image, GT = sample['rgb'], sample['gt']
    image = image.transpose( (2,0,1) )
    if len( GT.shape ) > 2:
      GT = numpy.transpose( GT, (2,0,1) )
    ret_dict = {'rgb':torch.from_numpy( image.copy() ).float(),
                'gt':torch.from_numpy( GT.copy() ).to( torch.int64 )}
    if 'boxes' in sample.keys():
      ret_dict['boxes'] = torch.as_tensor( sample['boxes'], dtype=torch.float32 )
    if 'labels' in sample.keys():
      ret_dict['labels'] = torch.as_tensor( sample['labels'], dtype=torch.int64 )
    if 'sub_labels' in sample.keys():
      ret_dict['sub_labels'] = torch.as_tensor( sample['sub_labels'], dtype=torch.int64 )
    if 'image_id' in sample.keys():
      ret_dict['image_id'] = torch.tensor( [sample['image_id']] )
    if 'cnt' in sample.keys():
      cnt = numpy.transpose( sample['cnt'], [2,0,1] ) if len( sample['cnt'].shape ) > 2 else sample['cnt']
      ret_dict['cnt'] = torch.from_numpy( cnt ).float()
    if 'cnt_selection' in sample.keys():
      cnt_selection = numpy.transpose( sample['cnt_selection'], [2,0,1] ) if len( sample['cnt_selection'].shape ) > 2 else sample['cnt_selection']
      ret_dict['cnt_selection'] = torch.from_numpy( cnt_selection ).float()
    if 'cnt_missing' in sample.keys():
      cnt_missing = numpy.transpose( sample['cnt_missing'], [2,0,1] ) if len( sample['cnt_missing'].shape ) > 2 else sample['cnt_missing']
      ret_dict['cnt_missing'] = torch.from_numpy( cnt_missing ).float()
    if 'cntcoords' in sample.keys():
      ret_dict['cntcoords'] = torch.as_tensor( sample['cntcoords'], dtype=torch.int64 )
    if 'reg' in sample.keys(): # this is a list
      reg = numpy.transpose( sample['reg'], [2,0,1] ) if len( sample['reg'].shape ) > 2 else sample['reg']
      ret_dict['reg'] = torch.from_numpy( reg ).float()
    for k in sample.keys():
      if k not in [ 'rgb', 'gt', 'boxes', 'labels', 'sub_labels', 'reg', 'cnt', 'cnt_selection', 'cnt_missing', 'cntcoords', ]:
        ret_dict[k] = sample[k]
    return ret_dict

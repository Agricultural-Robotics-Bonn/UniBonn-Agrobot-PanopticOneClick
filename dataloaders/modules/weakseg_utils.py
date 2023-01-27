

import numpy as np
from copy import deepcopy
import random
import cv2
from scipy.ndimage.morphology import binary_erosion



class WeakSegUtils():


  def __init__( self, ):
    pass


  def createClick( self,
                   gt,
                   keypoint,  # actual keypoint
                   noiserange,  # should noise be used
                 ):
    ##### creating clicks
    click_init = ( keypoint[1], keypoint[0] ) if keypoint is not None else self.generateCenterOfMassClick( gt )
    click = deepcopy(click_init)
    # add noise
    if noiserange is not None and noiserange > 0:
      # in case we use noise, there it is checked if it is valid, else returns (None,None) when 50 noise creation attempts fail
      click = self.addCenterNoise( gt, click_init, noiserange )
      if click[0] is None:
        # in that case, check if it was a keypoint click
        if keypoint is not None:
          # if so, try center of mass
          click = self.generateCenterOfMassClick( gt )
          click = self.addCenterNoise( gt, click, noiserange )
          # check again
          if click[0] is None:
            # if failed again do erosion, which is always valid
            click = self.generateErosionClick( gt )
            click = self.addCenterNoise( gt, click, noiserange, is_erosion_click=True )
        # if it wasnt a keypoint click, go straight to erosion as center of mass was already used initially
        else:
          click = self.generateErosionClick( gt )
          click = self.addCenterNoise( gt, click, noiserange, is_erosion_click=True )
    else:
      valid = self.checkClickValidity( click, gt )
      if not valid:
        click = self.generateCenterOfMassClick( gt )
        valid = self.checkClickValidity( click, gt )
        if not valid:
          click = self.generateErosionClick( gt )
    return click


  def addCenterNoise( self, gt, click, noiserange, is_erosion_click=False ):
    #### add noise randomization stuff
    if not isinstance( gt, np.ndarray ):
      gt = gt.numpy()
    # get non-zero indices
    cnt = 0
    for i in range(50):  # could be replaced by previously generating valid clicks and selecting from them, i.e., crop image (round crop?) -> nonzero -> select.
      step = random.randint( -noiserange, noiserange )
      cY_n = click[0] + step
      cX_n = click[1] + step
      cnt += 1
      if ( cY_n>=0 and cY_n<gt.shape[0] and cX_n>=0 and cX_n<gt.shape[1] ):  # just double checking if click is inside image, otherwise indexing in next line will fail
        if gt[cY_n,cX_n]==1:
          cX = cX_n
          cY = cY_n
          break
      # if didn't find a point matchin in gt after 50 iters, break if point is at least inside image
      if i==49:
        # there could be rare cases of very small objects where 50 iters of random noise might fail to get back to the object. As erosion is always valid, just take that one
        if is_erosion_click:
          cX = click[1]
          cY = click[0]
        else:
          cX = None
          cY = None
        break
    click_new = (cY,cX)
    return click_new


  def generateCenterOfMassClick( self, gt ):
    if not isinstance( gt, np.ndarray ):
      gt = gt.numpy()
    # calculate moments of binary image
    M = cv2.moments( gt, binaryImage=True )
    # calculate x,y coordinate of center
    if M["m00"] == 0:
      M["m00"] = .000000000000001  # experienced ZeroDivisionError at some point
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    center = (cY, cX)
    return center


  def generateErosionClick( self, gt ):
    '''
      determines click position based on the last leftover pixel(s) when completely eroding the mask
      if there are multiple ones in last iteration, a random one is chosen
    '''
    if not isinstance( gt, np.ndarray ):
      gt = gt.numpy()
    while gt.sum() > 0:
      nz = gt.nonzero()
      gt = binary_erosion( gt, iterations=1 )
    i = 0
    click = ( nz[0][i], nz[1][i] )
    return click


  def checkClickValidity( self, click, gt ):
    '''
      click has to be format (Y,Z)
    '''
    if not isinstance( gt, np.ndarray ):
      gt = gt.numpy()
    y = click[0]
    x = click[1]
    if ( y>=0 and y<gt.shape[0] and x>=0 and x<gt.shape[1] ):  # just double checking if click is inside image, otherwise indexing in next line will fail
      return True if gt[y,x]==1 else False
    else:
      return False




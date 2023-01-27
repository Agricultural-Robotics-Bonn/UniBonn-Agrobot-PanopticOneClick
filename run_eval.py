
import argparse
import yaml

import torch
from pytorch_lightning import Trainer

from dataloaders.coco import Parser as CocoLoader
from models.UNet_panoptic.module import PLModule as UNetModule



# Parse the command line arguments
def parsecmdline():
  parser = argparse.ArgumentParser( description='Extracting command line arguments', add_help=True )

  # config file, don't forget to include the path to config or another location if you change it.
  parser.add_argument( '--config', action='store' )
  parser.add_argument( '--gpus', '-g', action='store', type=int, default=1 )
  # what layer of verbositz are we using?
  parser.add_argument( "-v", "--verbosity", action="count", default=0 )
  parser.add_argument( "--prototyping", type=bool, default=False )

  ### own stuff
  parser.add_argument( '--model', action='store' )

  return parser.parse_args()



# The main function
def main( flags, ):

  ### own stuff
  if flags.model is not None and flags.config is None:
    flags.config = '/'.join( flags.model.split('/')[:-2] ) + '/config.yml'

  ##################################
  # load the config file
  with open( flags.config, 'r' ) as fid:
    cfg = yaml.load( fid, Loader=yaml.FullLoader )

  ### do we have a model? -> use it!
  if flags.model is not None:
    cfg['network']['pretrained'] = True
    cfg['network']['pretrained_path'] = flags.model

  ##################################
  # data loader
  data = CocoLoader( cfg )

  ##################################
  # data loader
  model = UNetModule( cfg )
  
  # load pretrained weights for eval
  if flags.verbosity > 1:
    print( 'loading a pretrained network:', cfg['network']['pretrained_path'] )
  pretrain = torch.load( cfg['network']['pretrained_path'], map_location='cpu' )
  model.load_state_dict( pretrain['state_dict'], strict=False )

  ##################################
  # trainer
  trainer = Trainer(gpus=flags.gpus,
                    accelerator="ddp" if flags.gpus > 1 else None,

                    max_epochs= cfg['trainer']['max_epochs'],

                    check_val_every_n_epoch=cfg['trainer']['val_every_n_epochs'],
                    precision=cfg['trainer']['precision'] if 'precision' in cfg['trainer'].keys() else 32,
                    resume_from_checkpoint=cfg['trainer']['checkpoint'] if 'checkpoint' in cfg['trainer'].keys() else None,

                    logger=[],
                    callbacks=[],
                    )

  # c. Run the trainer
  trainer.test( model, test_dataloaders=data.test_dataloader() )



if __name__ == '__main__':
  # parse the arguments
  flags = parsecmdline()

  # call the main function
  main( flags )

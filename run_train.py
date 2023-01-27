

import argparse
from datetime import datetime
import os
import yaml

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from dataloaders.coco import Parser as CocoLoader
from models.UNet_panoptic.module import PLModule as UNetModule



# Parse the command line arguments
def parsecmdline():
  parser = argparse.ArgumentParser( description='Extracting command line arguments', add_help=True )

  # config file, don't forget to include the path to config or another location if you change it.
  parser.add_argument( '--config', action='store', default='configs/min_example_config.yml' )
  parser.add_argument( '--gpus', '-g', action='store', type=int, default=1 )
  # what layer of verbositz are we using?
  parser.add_argument( "-v", "--verbosity", action="count", default=0 )
  parser.add_argument( "--prototyping", type=bool, default=False )

  return parser.parse_args()



# The main function
def main( flags, ):

  ##################################
  # load the config file
  with open( flags.config, 'r' ) as fid:
    cfg = yaml.load( fid, Loader=yaml.FullLoader )

  # other preamble
  callbacks = []
  timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
  cfg_name = flags.config.split('/')[-1].split('.')[0]
  experiment_name = f'{cfg["network"]["model"]}_train_{timestamp}--{cfg_name}'

  ##################################
  # data loader
  data = CocoLoader( cfg )

  ##################################
  # data loader
  model = UNetModule( cfg )

  # c. Load pretrained weights if required
  if cfg['network']['pretrained']:
    if flags.verbosity > 1:
      print( 'loading a pretrained network:', cfg['network']['pretrained_path'] )
    pretrain = torch.load( cfg['network']['pretrained_path'], map_location='cpu' )
    model.load_state_dict( pretrain['state_dict'], strict=False )

  ##################################
  # checkpoints
  # 4a. Save model with best val_loss
  if cfg['trainer']['checkpoints']['enable']:
    checkpoints_path = os.path.join( cfg['trainer']['checkpoints']['path'],
                                  experiment_name, 'checkpoints' )

    if 'every_n_val_epochs' in cfg['trainer']['checkpoints'].keys():
      callbacks.append( ModelCheckpoint( period=cfg['trainer']['checkpoints']['every_n_val_epochs'], # * cfg['trainer']['val_every_n_epochs'], # not sure what's happening here... "period"
                      dirpath=checkpoints_path,
                      save_top_k=-1,
                      filename='checkpoint_epoch_{epoch:04d}' ) )

  ##################################
  # trainer
  print('torch count:', torch.cuda.device_count())
  trainer = Trainer(gpus=flags.gpus,
                    accelerator="ddp" if flags.gpus > 1 else None,
                    max_epochs= cfg['trainer']['max_epochs'],
                    check_val_every_n_epoch=cfg['trainer']['val_every_n_epochs'],
                    precision=cfg['trainer']['precision'] if 'precision' in cfg['trainer'].keys() else 32,
                    logger=[],
                    callbacks=callbacks,
                    )

  # b. Save config file in log directory now that the trainer is configured
  # and ready to run. Save it using the active configuration dictionary. Why?
  # Because maybe the options at the top changed your configuration and you
  # want to be able to see what was actually used for this instance.
  #
  if cfg['logger']['log_cfg_file']:
    cfg_log_path = os.path.join( cfg['logger']['log_path'], experiment_name )
    if not os.path.isdir( cfg_log_path ):
      os.makedirs( cfg_log_path )
    with open( os.path.join( cfg_log_path, 'config.yml' ), 'w' ) as fid:
      yaml.dump( cfg, fid, default_flow_style=False, sort_keys=False )

  # c. Run the trainer
  trainer.fit( model, train_dataloader=data.train_dataloader(), val_dataloaders=data.val_dataloader() )



if __name__ == '__main__':
  # parse the arguments
  flags = parsecmdline()

  # call the main function
  main( flags )



import torch
from torch import nn
import torch.nn.functional as F



##########
class UNet_v2(nn.Module):

  ############
  def __init__(self, dropout=0.1, input_size=3, output_channels=2,
              backbone_filters=[64,128,256,512], bottleneck_filters=1024,
              head_filters=[512,256,128,64], max_pool_size=2, non_linearity="relu", **kwargs):
    """
    """

    print("UNet_v2::init Initilisation is ignoring the following arguments.", kwargs)

    assert (len(backbone_filters) == len(head_filters))

    # Initialisation of the basic attributes
    super(UNet_v2, self).__init__()
    self.dropout_prob = dropout
    self.input_size = input_size
    self.max_pool_size = max_pool_size
    self.non_linearity = non_linearity

    # Go through and initialise the attributes for the backbone(s)
    self.bbone_filters = backbone_filters
    self.down_sample_rates = []
    self.bbone_dict = {}
    self.bbone_modules = torch.nn.ModuleList()
    self.bbone_res = {}

    # Go through and initialise the attributes for the bottleneck(s)
    self.bneck_filters = bottleneck_filters
    self.bneck_dict = {}
    self.bneck_modules = torch.nn.ModuleList()
    self.bneck_res = {}

    # Go through and initialise the attributes for the head(s)
    self.head_filters = head_filters
    self.output_channels = output_channels
    self.head_dict = {}
    self.head_modules = torch.nn.ModuleList()
    self.head_res = {}

    # Make the network, backbone, bottleneck and then the head
    self.create_backbone()
    self.create_bottleneck()
    self.create_head()


  def choose_non_lienarity(self):
    """
    A function to return the chosen non-lienarity function. This abstracts the
    choice of non-lienarity and basically hides the nasty if statements
    """


    if self.non_linearity == "relu":
      return torch.nn.ReLU()
    elif self.non_linearity == "sigmoid":
      return torch.nn.Sigmoid()
    elif self.non_linearity == "softsign":
      return torch.nn.SoftSign()
    elif self.non_linearity == "softplus":
      return torch.nn.Softplus()
    elif self.non_linearity == "mish":
      return torch.nn.Mish()
    elif self.non_linearity == "swish":
      return torch.nn.silu()
    elif self.non_linearity == "tanh":
      return torch.nn.Tanh()
    else: # Defaults to relu, just in case
      return torch.nn.ReLU()


  ############
  def contractive_block(self, in_channels, out_channels, kernel_size=3):
    """
    This function creates one contractive block that consists of:
      a) 1 conv layer, batchnorm, relu, dropout
      b) 1 conv layer, batchnorm, relu, dropout
    """
    block = torch.nn.Sequential(
                torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels,
                                out_channels=out_channels, padding=1),
                torch.nn.BatchNorm2d(out_channels),
                self.choose_non_lienarity(),
                torch.nn.Dropout(p=self.dropout_prob),

                torch.nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels,
                                out_channels=out_channels, padding=1),
                torch.nn.BatchNorm2d(out_channels),
                self.choose_non_lienarity(),
                torch.nn.Dropout(p=self.dropout_prob)
                )
    return block


  ############
  def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
    """
    This function creates one expansive block that consists of:
      a) 1 conv layer, batchnorm, relu, dropout
      b) 1 conv layer, batchnorm, relu, dropout
      c) 1 conv transpose (stride 2), dropout
    """
    block = torch.nn.Sequential(
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
                    torch.nn.BatchNorm2d(mid_channel),
                    self.choose_non_lienarity(),
                    torch.nn.Dropout(p=self.dropout_prob),

                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1),
                    torch.nn.BatchNorm2d(mid_channel),
                    self.choose_non_lienarity(),
                    torch.nn.Dropout(p=self.dropout_prob),

                    # Maybe learning the upsampling filter leads to severe overfitting for particular datasets?
                    torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                    torch.nn.Dropout(p=self.dropout_prob))
    return  block


  ############
  def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
    """
    This returns the final block which consists of:
      a) 1 conv layer, batch norm, relu, dropout
      b) 1 conv layer, batch norm, relu, dropout
      c) 1 conv layer, batch norm, relu, dropout
    """
    block = torch.nn.Sequential(
                torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
                torch.nn.BatchNorm2d(mid_channel),
                self.choose_non_lienarity(),
                torch.nn.Dropout(p=self.dropout_prob),

                torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1),
                torch.nn.BatchNorm2d(mid_channel),
                self.choose_non_lienarity(),
                torch.nn.Dropout(p=self.dropout_prob),

                torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
                torch.nn.BatchNorm2d(out_channels),
                self.choose_non_lienarity(),
                torch.nn.Dropout(p=self.dropout_prob))
    return  block


  ############
  def create_backbone(self):
    """
    The backbone generally consists of something like:
      a) contractive block followed by the contraction (e.g. maxpooling)
      b) contractive block followed by the contraction (e.g. maxpooling)
      c) contractive block followed by the contraction (e.g. maxpooling)
    """

    # Use the previous input filter size
    previous_filter = self.input_size
    sample_rate = 1

    for (i, filter_size) in enumerate(self.bbone_filters):
      # Contractive block
      sample_rate *= self.max_pool_size  # the current sampling rate from this block
      label = 'encode'+str(i)
      curr_filter = self.contractive_block(in_channels=previous_filter, out_channels=filter_size)
      curr_pool = torch.nn.MaxPool2d(kernel_size=self.max_pool_size)
      curr_block = torch.nn.ModuleDict({'filter': curr_filter, 'pool': curr_pool})
      self.bbone_modules.append(curr_block)
      self.bbone_dict[sample_rate] = {"label": label, "index_to_list": i, "filter_size": filter_size}
      previous_filter = filter_size
      self.down_sample_rates.append(sample_rate)


  ###########
  def create_bottleneck(self, bneck_kernel=3, upsample=2):
    """
    The bottleneck generally consists of something like:
      a) 1 conv layer, batchnorm, relu, dropout
      b) 1 conv layer, batchnorm, relu, dropout
      c) 1 conv transpose (stride 2) layer, dropout
    """

    # Grab the size of the final backbone layer filter and the head
    previous_filter = self.bbone_filters[-1]
    out_filter_size = self.head_filters[0]

    label = 'bneck_layer1'
    filter_size = self.bneck_filters
    bneck_layer = torch.nn.Sequential(
                        torch.nn.Conv2d(kernel_size=bneck_kernel, in_channels=previous_filter, out_channels=filter_size, padding=1),
                        torch.nn.BatchNorm2d(filter_size),
                        self.choose_non_lienarity(),
                        torch.nn.Dropout(p=self.dropout_prob),

                        torch.nn.Conv2d(kernel_size=bneck_kernel, in_channels=filter_size, out_channels=filter_size, padding=1),
                        torch.nn.BatchNorm2d(filter_size),
                        self.choose_non_lienarity(),
                        torch.nn.Dropout(p=self.dropout_prob),

                        torch.nn.ConvTranspose2d(in_channels=filter_size, out_channels=out_filter_size, kernel_size=bneck_kernel, stride=upsample, padding=1, output_padding=1),
                        torch.nn.Dropout(p=self.dropout_prob) )
    self.bneck_modules.append(bneck_layer)
    self.bneck_dict["1"] = {"label": label, "index_to_list": 0, "filter_size": filter_size}



  ##############
  def create_head(self):
    """
    The head layer generally consists of something like:
      a) bringing together skip connections, expansive block  [for loop]
      b) bringing together skip connections, expansive block  [for loop]
      c) bringing together skip connections, head block  [outside of for loop]
    """

    # Use the previous input filter size
    # This is how it was used to get the final layer size of the bottleneck layer
    previous_filter_size = self.head_filters[0]
    sample_rate = self.down_sample_rates[-1]

    for i in range(0, len(self.head_filters)-1):
      # Skip layer (from decoder)
      decoder_filter_size = self.bbone_dict[sample_rate]["filter_size"]
      in_filter_size = decoder_filter_size + previous_filter_size
      filter_size = self.head_filters[i]
      out_filter_size = self.head_filters[i+1]
      label = 'decode'+str(i+1)

      filters = self.expansive_block(in_filter_size, filter_size, out_filter_size)
      self.head_modules.append(filters)

      self.bbone_dict[sample_rate] = {"label": label, "index_to_list": i, "filter_size": filter_size} ###
      previous_filter_size = out_filter_size
      sample_rate //= self.max_pool_size  # the current sampling rate from this block


    # Now generate the final layer of the head node..
    decoder_filter_size = self.bbone_dict[sample_rate]["filter_size"]
    in_filter_size = decoder_filter_size + previous_filter_size
    filter_size = self.head_filters[-1]
    out_filter_size = self.output_channels
    self.final_layer = torch.nn.ModuleList()
    self.final_layer.append( self.final_block(in_filter_size, filter_size, out_filter_size) )



  #############
  def crop_and_concat(self, upsampled, bypass, crop=False):
    """
    This layer crop the layer from contraction block and concat it with expansive block vector
    """
    if crop:
      c = (bypass.size()[2] - upsampled.size()[2]) // 2
      bypass = F.pad(bypass, (-c, -c, -c, -c))
    return torch.cat((upsampled, bypass), 1)


  def calculate_smax(self, outputs):
    return torch.nn.functional.softmax(outputs, dim=1)  # B, C, H, W

  #############
  def forward(self, x):

    #### Pass through the encoding filters
    input_x = x
    sampling_rate = 1
    for i in range(len(self.bbone_modules)):
      sampling_rate *= 2
      filter_res = self.bbone_modules[i]['filter'](input_x)
      pool_res = self.bbone_modules[i]['pool'](filter_res)
      self.bbone_res[sampling_rate] = {'filter_res': filter_res, 'pool_res': pool_res}
      input_x = pool_res  # link back for the next layer

    #### Pass through the bottleneck filters
    for i in range(len(self.bneck_modules)):
      filter_res = self.bneck_modules[i](input_x)
      input_x = filter_res # link back for the next layer

    #### Pass through the decoding filters
    for i in range(len(self.head_modules)):
      ##### This has to be implemented properly...
      crop_cat_res = self.crop_and_concat(input_x, self.bbone_res[sampling_rate]['filter_res'], crop=False)
      sampling_rate //= 2
      filter_res = self.head_modules[i](crop_cat_res)
      input_x = filter_res  # link back for the next layer

    #### Get the output from the final layer
    self.final_results = []
    self.penultimate_input_x = input_x
    for i in range(len(self.final_layer)):
      ##### This has to be implemented properly...
      crop_cat_res = self.crop_and_concat(input_x, self.bbone_res[sampling_rate]['filter_res'], crop=False)
      self.final_results.append(self.final_layer[0](crop_cat_res))

    self.smax = self.calculate_smax(self.final_results[0])

    #### Return this.
    return {"out": self.final_results[0], "smax": self.smax}

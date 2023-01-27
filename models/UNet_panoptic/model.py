
import torch
import torch.nn.functional as F

from .UNet_v2 import UNet_v2



##########
class UNet_tag_point(UNet_v2):

  ############
  def __init__(self, dropout=0.1, input_size=3, 
               backbone_filters=[64,128,256,512], 
               bottleneck_filters=1024,
               output_channel_set=[2,2],
               head_filter_set=[[512,256,128,64], [512,256,128,64]], 
               head_types=['segmentation','regression'],
               head_keys=['tag_segmentation','tag_regression'],
               max_pool_size=2, non_linearity="relu", **kwargs):
    """
    Make a set of of parameters for each of the different heads that we have 
    and also for the corresponding output channels.
    """

    print("UNet_tag_point::init Initilisation is ignoring the following arguments.", kwargs)


    assert (len(head_filter_set) == len(head_types) )
    assert (len(head_filter_set) == len(head_keys) )
    assert (len(head_filter_set) == len(output_channel_set) )

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

    # Go through and initialise the attributes for the head(s)
    self.head_filter_set = head_filter_set
    self.output_channel_set = output_channel_set
    self.head_types = head_types
    self.head_keys = head_keys
    self.head_dict = torch.nn.ModuleDict()  # The dictionary to hold the different heads of interest for this model

    # Make the network, backbone, bottleneck and then the head
    self.create_backbone()
    self.create_bottleneck()
    self.create_multiple_heads()


  def choose_non_lienarity(self, non_linearity=None):
    """
    A function to return the chosen non-lienarity function. This abstracts the
    choice of non-lienarity and basically hides the nasty if statements
    """

    if non_linearity==None:
      non_linearity = self.non_linearity

    if non_linearity == "relu":
      return torch.nn.ReLU()
    elif non_linearity == "sigmoid":
      return torch.nn.Sigmoid()
    elif non_linearity == "softsign":
      return torch.nn.SoftSign()
    elif non_linearity == "softplus":
      return torch.nn.Softplus()
    elif non_linearity == "mish":
      return torch.nn.Mish()
    elif non_linearity == "swish":
      return torch.nn.silu()
    elif non_linearity == "tanh":
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
  def final_regression_block(self, in_channels, mid_channel, out_channels, kernel_size=3, non_linearity=None):
    """
    This returns the final block which consists of:
      a) 1 conv layer, batch norm, relu, dropout
      b) 1 conv layer, batch norm, relu, dropout
      c) 1 conv layer, batch norm, relu, dropout
    """
    block = torch.nn.Sequential(
                torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
                self.choose_non_lienarity(),

                torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1),
                self.choose_non_lienarity(),

                torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1))
                # No non-linearity at the end
    return  block

  ############
  def final_center_block(self, in_channels, mid_channel, out_channels, kernel_size=3, non_linearity=None):
    """
    This returns the final block which consists of:
      a) 1 conv layer, batch norm, relu, dropout
      b) 1 conv layer, batch norm, relu, dropout
      c) 1 conv layer, batch norm, relu, dropout
    """
    block = torch.nn.Sequential(
                torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
                self.choose_non_lienarity(),

                torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1),
                self.choose_non_lienarity(),

                torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
                self.choose_non_lienarity('sigmoid')
                )
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
    out_filter_size = self.head_filter_set[0][0]

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
  def create_multiple_heads(self):
    """
    Taking the set and creating a head for each set.
    """

    #### Go through and get the set of data to use...
    for c_type, c_key, head_filters, output_channels in zip(self.head_types,
                                                            self.head_keys, 
                                                            self.head_filter_set, 
                                                            self.output_channel_set):

      if c_type == "segmentation":
        curr_module_list, curr_final_layer = self.create_segmentation_head(head_filters, output_channels)
      elif c_type == "regression":
        curr_module_list, curr_final_layer = self.create_regression_head(head_filters, output_channels)
      elif c_type == "center_regression":
        curr_module_list, curr_final_layer = self.create_regression_head(head_filters, output_channels, is_center_head=True)
      else:
        print("UNet_tag_point::create_multiple_heads Invalid head type chosen", c_type)
        break

      self.head_dict[c_key] = torch.nn.ModuleDict({'c_modules': curr_module_list, 'c_final_layer': curr_final_layer})





  ##############
  def create_segmentation_head(self, head_filters, output_channels):
    """
    The head layer generally consists of something like:
      a) bringing together skip connections, expansive block  [for loop]
      b) bringing together skip connections, expansive block  [for loop]
      c) bringing together skip connections, head block  [outside of for loop]
    """

    # Use the previous input filter size
    # This is how it was used to get the final layer size of the bottleneck layer
    curr_module_list = torch.nn.ModuleList()
    previous_filter_size = head_filters[0]
    sample_rate = self.down_sample_rates[-1]

    for i in range(0, len(head_filters)-1):
      # Skip layer (from decoder)
      decoder_filter_size = self.bbone_dict[sample_rate]["filter_size"]
      in_filter_size = decoder_filter_size + previous_filter_size
      filter_size = head_filters[i]
      out_filter_size = head_filters[i+1]
      label = 'decode'+str(i+1)

      filters = self.expansive_block(in_filter_size, filter_size, out_filter_size)
      curr_module_list.append(filters)

      self.bbone_dict[sample_rate] = {"label": label, "index_to_list": i, "filter_size": filter_size} ###
      previous_filter_size = out_filter_size
      sample_rate //= self.max_pool_size  # the current sampling rate from this block


    # Now generate the final layer of the head node..
    decoder_filter_size = self.bbone_dict[sample_rate]["filter_size"]
    in_filter_size = decoder_filter_size + previous_filter_size
    filter_size = head_filters[-1]
    out_filter_size = output_channels
    curr_final_layer = torch.nn.ModuleList()
    curr_final_layer.append( self.final_block(in_filter_size, filter_size, out_filter_size) )

    return curr_module_list, curr_final_layer


  ##############
  def create_regression_head(self, head_filters, output_channels, is_center_head=False):
    """
    The head layer generally consists of something like:
      a) bringing together skip connections, expansive block  [for loop]
      b) bringing together skip connections, expansive block  [for loop]
      c) bringing together skip connections, head block  [outside of for loop]
    """

    # Use the previous input filter size
    # This is how it was used to get the final layer size of the bottleneck layer
    curr_module_list = torch.nn.ModuleList()
    previous_filter_size = head_filters[0]
    sample_rate = self.down_sample_rates[-1]

    for i in range(0, len(head_filters)-1):
      # Skip layer (from decoder)
      decoder_filter_size = self.bbone_dict[sample_rate]["filter_size"]
      in_filter_size = decoder_filter_size + previous_filter_size
      filter_size = head_filters[i]
      out_filter_size = head_filters[i+1]
      label = 'decode'+str(i+1)

      filters = self.expansive_block(in_filter_size, filter_size, out_filter_size)
      curr_module_list.append(filters)

      self.bbone_dict[sample_rate] = {"label": label, "index_to_list": i, "filter_size": filter_size} ###
      previous_filter_size = out_filter_size
      sample_rate //= self.max_pool_size  # the current sampling rate from this block


    # Now generate the final layer of the head node..
    decoder_filter_size = self.bbone_dict[sample_rate]["filter_size"]
    in_filter_size = decoder_filter_size + previous_filter_size
    filter_size = head_filters[-1]
    out_filter_size = output_channels
    curr_final_layer = torch.nn.ModuleList()
    if is_center_head:
      curr_final_layer.append( self.final_center_block(in_filter_size, filter_size, out_filter_size) )
    else:
      curr_final_layer.append( self.final_regression_block(in_filter_size, filter_size, out_filter_size) )
    return curr_module_list, curr_final_layer


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


    #### Pass through each head in order..
    final_results = {}
    START_sampling_rate = sampling_rate # The sampling rate at this point.
    INITIAL_input = input_x # The input, from the bottleneck layer, at this point.
    for c_key in self.head_keys:
      sampling_rate = START_sampling_rate
      input_x = INITIAL_input
      c_head_module = self.head_dict[c_key]['c_modules']
      c_final_layer = self.head_dict[c_key]['c_final_layer']

      #### Pass through the decoding filters
      for i in range(len(c_head_module)):
        ##### This has to be implemented properly...
        crop_cat_res = self.crop_and_concat(input_x, self.bbone_res[sampling_rate]['filter_res'], crop=False)
        sampling_rate //= 2
        filter_res = c_head_module[i](crop_cat_res)
        input_x = filter_res  # link back for the next layer

      #### Get the output from the final layer
      c_final_results = []
      c_penultimate_input_x = input_x
      for i in range(len(c_final_layer)):
        ##### This has to be implemented properly...
        crop_cat_res = self.crop_and_concat(input_x, self.bbone_res[sampling_rate]['filter_res'], crop=False)
        c_final_results.append(c_final_layer[0](crop_cat_res))
      assert len(c_final_results)==1, 'Exited for safety.. Next step was modified to make panoptic one-click based on Unet work with our panoptic-deeplab implementation. Double check if this is still doing as desired when using for different purpose.'
      final_results[c_key] = c_final_results[0]  # modified to work with our panoptic-deeplab implementation.

    #### Return this.
    return final_results










"""
  Helper functions for the dataloaders
"""

# a collate function to handle different lengths in dataloader elements
# using this to be able to parse click locations...
def collate_variable_masks( batch ):
  output = {}
  # iterate over the batches
  for b in batch:
    for k, v in b.items():
      try:
        output[k].append( v )
      except:
        output[k] = [v]
  # return the dictionary of lists
  return output

# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd 
import numpy as np
import random
from transformers import AutoModelForTokenClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import time
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import torch


# All the hyperparameters for fine-tuning
MODEL_CHECKPOINT = "bert-base-uncased"
BATCH_SIZE = 2
LEARNING_RATE= 5e-5    
NUM_TRAIN_EPOCHS= 1
EPS = 1e-8
SEED = 4612

# this function: train the model
def train_model(train_dataloader):

  # Set the seed value all over the place to make this reproducible.
  random.seed(SEED)
  np.random.seed(SEED)
  torch.manual_seed(SEED)
  torch.cuda.manual_seed_all(SEED)

  model = AutoModelForTokenClassification.from_pretrained(MODEL_CHECKPOINT, num_labels = 3 + 1, output_attentions = False,  output_hidden_states = False)
  device = torch.device("mps")
  model.to(device)
  # Load the AdamW optimizer
  optimizer = AdamW(model.parameters(), lr = LEARNING_RATE, eps = EPS)
  # Total number of training steps is number of batches * number of epochs.
  total_steps = len(train_dataloader) * NUM_TRAIN_EPOCHS
  # Create the learning rate scheduler.
  scheduler = get_linear_schedule_with_warmup(optimizer,  num_warmup_steps = 0, num_training_steps = total_steps)
  def format_time(elapsed):
    #Takes a time in seconds and returns a string hh:mm:ss
    elapsed_rounded = int(round((elapsed)))   
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
  loss_values = []
  # For each epoch...
  for epoch_i in range(0, NUM_TRAIN_EPOCHS):     
      print("")
      print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, NUM_TRAIN_EPOCHS))
      print('Training...')
      t0 = time.time()
      # Reset the total loss for this epoch.
      total_loss = 0
      model.train()

      for step, batch in enumerate(train_dataloader):
          if step % 40 == 0 and not step == 0:
              elapsed = format_time(time.time() - t0)
              print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

          b_input_ids = batch[0].to(device)
          b_input_mask = batch[1].to(device)
          b_labels = batch[2].to(device)

          # clear any previously calculated gradients before performing a backward pass.
          model.zero_grad()        

          # Perform a forward pass
          outputs = model(b_input_ids,token_type_ids=None,  attention_mask=b_input_mask, labels=b_labels)          
          loss = outputs[0]
          # `loss` is a Tensor containing a single value.'.item()' function just returns the Python value 
          total_loss += loss.item()
          # Perform a backward pass to calculate the gradients.
          loss.backward()
          # Clip the norm of the gradients to 1.0, helping prevent the "exploding gradients" problem.
          torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
          # Update parameters and take a step using the computed gradient.
          optimizer.step()
          # Update the learning rate.
          scheduler.step()

      # Calculate the average loss over all batches of size = BATCH_SIZE for one epoch
      avg_train_loss = total_loss / len(train_dataloader)                  
      # Store the loss value for each epoch for plotting the learning curve.
      loss_values.append(avg_train_loss)
      print("")
      print("  Average training loss: {0:.2f}".format(avg_train_loss))
      print("  Training epoch took: {:}".format(format_time(time.time() - t0)))
            
  print("")
  print("Training complete!")
  # Use plot styling from seaborn.
  sns.set(style='darkgrid')
  # Increase the plot size and font size.
  sns.set(font_scale=1.5)
  plt.rcParams["figure.figsize"] = (12,6)
  # Plot the learning curve.
  plt.plot(loss_values, 'b-o')
  # Label the plot.
  plt.title("Training loss")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.show()
  
  return model
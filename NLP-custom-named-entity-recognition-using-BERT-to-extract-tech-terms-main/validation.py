# -*- coding: utf-8 -*-

from sklearn.metrics import f1_score,accuracy_score,confusion_matrix
import numpy as np
import torch

# this function: Evaluate the model
def validate_model(model, dataloader, tags):
    print('Predicting labels for {:,} sentences...'.format(len(tags)))
    # Put model in evaluation mode
    model.eval()
    # Tracking variables 
    predictions , true_labels = [], []
    # Predict 
    for batch in dataloader:
      # Add batch to GPU
      device = torch.device("mps")
      batch = tuple(t.to(device) for t in batch)
      b_input_ids, b_input_mask, b_labels = batch     
      # not to compute or store gradients
      with torch.no_grad():
          outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
      logits = outputs[0]
      # Move logits and labels to CPU
      logits = logits.detach().cpu().numpy()
      label_ids = b_labels.to('cpu').numpy()
      # Store predictions and true labels
      predictions.append(logits)
      true_labels.append(label_ids)
    print('    DONE.')

    all_predictions = np.concatenate(predictions, axis=0)
    all_true_labels = np.concatenate(true_labels, axis=0)
    print("After flattening the batches, the predictions have shape:")
    print("    ", all_predictions.shape)
    predicted_label_ids = np.argmax(all_predictions, axis=2)
    print("\nAfter choosing the highest scoring label for each token:")
    print("    ", predicted_label_ids.shape) 
    predicted_label_ids = np.concatenate(predicted_label_ids, axis=0)
    all_true_labels = np.concatenate(all_true_labels, axis=0)
    print("\nAfter flattening the sentences, we have predictions:")
    print("    ", predicted_label_ids.shape)
    print("and ground truth:")
    print("    ", all_true_labels.shape)

    # Construct new lists of predictions which don't include any null tokens.
    real_token_predictions = []
    real_token_labels = []

    for i in range(len(all_true_labels)):
        if not all_true_labels[i] == -100:
            real_token_predictions.append(predicted_label_ids[i])
            real_token_labels.append(all_true_labels[i])
    print("Before filtering out `null` tokens, length = {:,}".format(len(all_true_labels)))
    print(" After filtering out `null` tokens, length = {:,}".format(len(real_token_labels)))

    validation_f1 = f1_score(real_token_labels, real_token_predictions,average = None) #, average='micro'
    validation_accuracy = accuracy_score(real_token_labels, real_token_predictions) 
    print ("Overall accuracy: {:.2%}".format(validation_accuracy))
    print("F1 score in O, I and B class:" , validation_f1)
    print("confusion matrix for O, I, B: ")
    print(confusion_matrix(real_token_labels, real_token_predictions))
    return
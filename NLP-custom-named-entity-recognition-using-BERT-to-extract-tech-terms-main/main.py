# import functions from the following modules:
# Please read the modules themselves for details.
from clean import load_and_clean_data
from preprocess import preprocess_data_for_bert
from train import train_model
from validation import validate_model
from extractor import detect_tech_terms_in_articles_csv
import os

# decide the data we will use for training and testing
train = ["m1","inventions","AI","aircraft","bio_med_health","blockchain","cloning","emerging","genetic","neuro","p2p","quantum","robotics","routing","transport","wireless", "train_data_added0","train_data_added1","train_noise_added","train_data_added3","train_data_added4"]
test = ['test']

# set up all the hyperparameters for fine-tuning BERT
MODEL_CHECKPOINT = "bert-base-uncased"
BATCH_SIZE = 8
LEARNING_RATE= 5e-5
NUM_TRAIN_EPOCHS= 7
EPS = 1e-8
SEED = 4612

# the output_dir where we save the model and tokenizers
output_dir = './saved_bert_model_and_tokenizer_m2_v12/'

# choose a csv file with new articles where you want to extract technology terms
# the data we use here are AI related articles from train set as an example
df = "data/all_untagged_data/train/AI.csv"


def main():
    '''
    Pipeline for
    1. cleaning & preprocessing train and test sets,
    2. fine-tuning the BERT model with train set,
    3. validating the fine-tuned BERT model performance for both train and test set,
    4. save the fune-fined BERT model and tokenizer,
    5. use the fune-fined BERT model and tokenizer for inference (identify tech terms and extract them from any articles)
    '''

    # load and cleaning train and test data (sentences and tags)
    train_sentences, train_tags = load_and_clean_data(train)
    test_sentences, test_tags = load_and_clean_data(test)

    # preprocess the clean train and test data for BERT
    train_dataloader, tokenizer = preprocess_data_for_bert(train_sentences, train_tags, False, True)
    test_dataloader, tokenizer = preprocess_data_for_bert(test_sentences, test_tags, False, False)

    # train the model
    model = train_model(train_dataloader)

    # calculate F1 score, confusion matrix for train set
    validate_model(model, train_dataloader, train_tags)

    # calculate F1 score, confusion matrix for test set
    validate_model(model, test_dataloader, test_tags)

    # save the model and tokenizer so we can use them for inference (extractor.py)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("Saving model to %s" % output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # use the saved model to extract technology terms from new data
    df_test = detect_tech_terms_in_articles_csv(df, output_dir)
    # save the prediction results to a csv file if you'd like to. Change the directory as needed.
    df_test.to_csv('data/all_untagged_data/AI_tech_terms.csv', index=False, encoding='utf-8-sig')

if __name__ == '__main__':
    main()


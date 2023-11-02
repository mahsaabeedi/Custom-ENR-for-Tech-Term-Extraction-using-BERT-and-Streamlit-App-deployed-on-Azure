# Custom ENR for Tech Term Extraction using BERT and Streamlit App deployed on Azure  


## Project Highlights

I successfully developed and delivered a BERT-based model that has undergone fine-tuning, despite utilizing a relatively small dataset. This model is specifically designed to identify and extract technology-related terms from various articles, achieving impressively high F1 scores consistently exceeding 90%.

The development process involved constructing a comprehensive pipeline consisting of the following key stages:

* **Data Cleaning and Preprocessing**: The first step in the pipeline was the meticulous cleaning and preprocessing of the training and testing data. This stage ensured that the data fed into the BERT model was well-structured and devoid of any inconsistencies or noise, optimizing the model's performance.

* **Fine-Tuning BERT Model**: Once the data was prepared, the BERT model was fine-tuned using the training set. Fine-tuning is a critical phase where the model learns to recognize and classify technology terms by adapting to the specific characteristics of the provided data. This process allows the model to improve its ability to identify relevant terms accurately.

* **Performance Validation**: After fine-tuning, the model's performance was thoroughly evaluated. This involved calculating the F1 score, a common metric used to assess the model's precision and recall. Additionally, a confusion table was generated to provide insights into the model's predictions and misclassifications. The evaluation was conducted for both the training and testing datasets, ensuring that the model generalizes well and does not overfit.

* **Model and Tokenizer Saving**: Upon achieving satisfactory results, the fine-tuned BERT model and its associated tokenizer were saved. This step is crucial for ensuring that the model can be readily deployed and used for future tasks without the need for retraining.

* **Inference and Term Extraction**: The final stage in the pipeline involved deploying the fine-tuned BERT model and its tokenizer for inference on new, unseen articles. When presented with a fresh article, the model applies its learned knowledge to identify and extract technology-related terms effectively. 

## Data Cleaning

* Reads tagged data from text files (B, I, O tags).
* Cleans and separates tags from mixed tags (e.g., "B-technology" to "B").
* Converts the data into a list of lists format, where each list represents a sentence and its corresponding tags.
* Combines data from multiple files if provided as a list of file names.
* Ensures the lengths of token and tag sequences are equal for each example.
* Returns two lists: one for sentences and another for tags, ready for further processing or training in sequence labeling tasks.

## Code and References

* Language: Python 3.9 and PyTorch
* BERT Research: from Chris McCormick https://www.chrismccormick.ai/
* BERT Paper: https://arxiv.org/abs/1810.04805
* Transformers Docs: https://huggingface.co/transformers/
* Transformers Repo: https://github.com/huggingface/transformers
* Packages Used: tensorflow, torch, numpy, pandas , seaborn, matplotlib, google.colab, sklearn, transformers, time, datetime, random, os , tokenizers

## Model Training and Fine-tuning
The code is structured to perform fine-tuning on a custom dataset with specified hyperparameters:

* Defines a function train_model to perform the training of the model.
* Initializes the BERT-based model for token classification with the specified configuration.
* Defines the device for training (e.g., GPU) and moves the model to that device.
* Initializes the AdamW optimizer for model parameter updates.
* Creates a learning rate scheduler for a linear warm-up.
* Initializes an empty list loss_values to store training loss values.

* Hyperparameters:
  * MODEL_CHECKPOINT: Specifies the BERT model checkpoint to use (e.g., "bert-base-uncased").
  * BATCH_SIZE: Sets the batch size for training.
  * LEARNING_RATE: Defines the learning rate for optimization.
  * NUM_TRAIN_EPOCHS: Specifies the number of training epochs.
  * EPS: A small constant to prevent division by zero in optimization.
  * SEED: A seed value for reproducibility.

![Alt text](https://github.com/mahsaabeedi/Custom-ENR-for-Tech-Term-Extraction-using-BERT-and-Streamlit-App-deployed-on-Azure/blob/main/NLP-custom-named-entity-recognition-using-BERT-to-extract-tech-terms-main/Figure_1.png)

## Model Performance

I delivered a BERT-based model that was fine-tuned with a small volume of data to extract all technology terms from any given article (F1 scores > 90%)
The results can be derived from running these 2 commands:
* Install dependencies with $ pip install -r requirements.txt
* Run the model with $ python3 main.py

## Deployment of Technology Term Extractor Web App Using Streamlit and Deployed with Docker on Azure

I deployed the fine-tuned BERT model as a web app with Docker in Azure. The app can help users extract technology terms from articles in multiple ways. Please find the code for the web app deployment in this [repo](https://github.com/mahsaabeedi/Custom-ENR-for-Tech-Term-Extraction-using-BERT-and-Streamlit-App-deployed-on-Azure/tree/main/techterm_extractor_using_app_streamlit_deployed_on_azure).

This repository contains a Dockerized Python application that leverages the BERT model and Streamlit for interactive web-based data analysis. It's designed to be easy to set up and deploy in a containerized environment. The application is built on top of Python 3.9 and uses the Transformers library for BERT support.

# Dockerized Python Application with BERT and Streamlit

Welcome to my GitHub repository! This project demonstrates a Dockerized Python application that utilizes the BERT model and Streamlit for creating an interactive web-based data analysis tool. The application is designed for easy setup and deployment within a containerized environment. It's built on Python 3.9 and uses the Transformers library for BERT support.

## Features

- Interactive web-based interface using Streamlit.
- Incorporation of a BERT model for various natural language processing tasks.
- Docker container for straightforward deployment.

## Prerequisites

Before you get started, make sure you have the following prerequisites installed:

- Docker: [Get Docker](https://docs.docker.com/get-docker/)
- Python 3.9
- Required Python packages as listed in `requirements.txt`

## Building and Running the Docker Container

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
2. Build the Docker container:
   
   ```bash
   docker build -t techdetector:v1 .
3. Run the Docker container:

   ```bash
    docker run -d --name mywebapp -p 8501:8501 techdetector:v1
5. Access the application by visiting http://localhost:8501 in your web browser. After reviewing the app which is working fine, I pushed the image to ACR.
6. Pushing to Azure Container Registry
To push my Docker image to an Azure Container Registry (ACR) for cloud deployment, I can follow these steps:
* Ensure I have an Azure account and am logged in via the Azure CLI.
* Log in to my Azure Container Registry using the Azure CLI:
    ```bash
     az acr login --name my-acr-name
* Tag my Docker image to match my ACR login server:
    ```bash
    docker tag my-image-name my-acr-name.azurecr.io/my-image-name
* Push my Docker image to the Azure Container Registry:
    ```bash
    docker push my-acr-name.azurecr.io/my-image-name

# Web Application Visualization
* The web app is here: https://tech-detector.azurewebsites.net/ (deleted to stop billing)
* The app can help you extract distinct technology terms from text. There are 3 options.
  * Option 1 (under 450 words): Enter a url. The app will return the scraped text from the url and the technology terms in the text.
  * Option 2 (under 450 words): Enter some text. The app will return the input text and the technology terms in the text.
  * Option 3 (for multiple articles and over 450 words): Upload a file with one column named "text". The app will return a csv file with the input articles and the detected technology terms for each article. You can download the returned file.


_Below is a screenshot of the app:_

![alt text](https://github.com/mahsaabeedi/Custom-ENR-for-Tech-Term-Extraction-using-BERT-and-Streamlit-App-deployed-on-Azure/blob/main/techterm_extractor_using_app_streamlit_deployed_on_azure/web%20app%20screenshot.png)


# Image-Caption-Generation

## Image Captioning with CNN + LSTM and ResNet + GRU

This repository explores two popular approaches for automatic image captioning: CNN + LSTM and ResNet + GRU. We use the Flickr 8k dataset and compare the performance of these approaches using BLEU scores.

### Prerequisites

This project requires the following libraries:

* Python 3.6+
* TensorFlow or PyTorch
* NumPy
* Pandas
* Matplotlib

### Data

The Flickr 8k dataset contains 8,092 images and 5 captions each. You can download the dataset from [https://www.kaggle.com/datasets/adityajn105/flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k).

### Notebooks

This repository contains two Jupyter notebooks:

* `CNN + LSTM`: Implements an image captioning model using a CNN encoder and an LSTM decoder.
* `Resnet + GRU`: Implements an image captioning model using a ResNet encoder and a GRU decoder.

Both notebooks demonstrate the following steps:

1. **Preprocessing the data:** Downloading the Flickr 8k dataset, loading the images and captions, and preprocessing the text data.
2. **Defining the model architecture:** Implementing the CNN + LSTM or ResNet + GRU architecture using TensorFlow or PyTorch.
3. **Training the model:** Training the model on the Flickr 8k dataset.
4. **Generating captions for images:** Using the trained model to generate captions for new images.
5. **Evaluating the model:** Evaluating the model's performance using BLEU scores.

### Performance Comparison

We will compare the performance of the CNN + LSTM and ResNet + GRU models using BLEU scores. BLEU score is a widely used metric for evaluating the quality of machine-generated text. It measures the similarity between the generated captions and the reference captions.

Here are some factors that can affect the performance of the models:

* **Hyperparameters:** The choice of hyperparameters, such as the learning rate and the number of hidden units, can have a significant impact on the model's performance.
* **Preprocessing:** The way the data is preprocessed can also affect the model's performance. For example, using different tokenization techniques or different vocabulary sizes can lead to different results.
* **Training data:** The quality and quantity of the training data can also impact the model's performance.

### Getting Started

1. Clone this repository.
2. Install the required libraries.
3. Download the Flickr 8k dataset.
4. Open the Jupyter notebooks and run the code.
5. Compare the performance of the two models using BLEU scores.


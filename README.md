# Word2doc2vec-Doc-relevance-training
This repository focuses on an approach exploring and evaluating literature-based document-to-document (doc-2-doc) recommendations based on the Word2Vec technique. The approach involves generating document-level embeddings through centroid aggregation of word embeddings. The dataset used is the RELISH Corpus, an expert-curated collection of biomedical literature consisting of pairwise document assessments. The workflow comprises of training Word2Vec models on a predefined training set, followed by the assessment of doc-2-doc recommendations on a distinct test set. Additionally, Optuna is utilized to optimize the hyperparameters of the trained Word2Vec models.

## 📚🔍Table of Contents

1. [About](#📝-about)
2. [Input Data](#📂input-data)
3. [Pipeline](#🛠️pipeline)
    1. [Train and Optimize Word2vec models](#🧠⚙️-train-and-optimize-word2vec-models)
    2. [Calculate Cosine Similarity](#📐🔄calculate-cosine-similarity)
    3. [Evaluation](#📈📋-evaluation)
        - [Precision@N](#🎯precisionn)
        - [nDCG@N](#📊-ndcgn)
4. [Code Implementation]()
5. [Getting Started](#🚀-getting-started)


## 📝 About

Our approach involves utilizing [Word2Vec](https://arxiv.org/pdf/1310.4546.pdf) for capturing word-level semantics and generating word embeddings. By averaging these word vectors, our centroid approach aggregates the individual word embeddings into a unified document-level representation, facilitating more effective analysis and comparison of documents based on their titles and abstracts.

## 📂Input Data

The input data for this method includes preprocessed tokens derived from the RELISH documents, a specialized database curated by experts for benchmarking document similarity in biomedical literature. The RELISH dataset comprises a JSON file containing PubMed IDs (PMIDs) along with document-to-document relevance assessments categorized as "relevant," "partial," or "irrelevant." Titles and abstracts of the associated articles were retrieved and stored in a TSV file. 

The title and abstract text are preprocessed, and the resulting tokens are stored in the RELISH.npy file, which includes arrays of PMIDs, document titles, and abstracts. These arrays are produced through an extensive preprocessing pipeline, detailed in the [relish-preprocessing repository](https://github.com/zbmed-semtec/relish-preprocessing). This pipeline includes several refinement stages for both titles and abstracts: structural words are removed, text is converted to lowercase, and tokenization is applied to create arrays of individual words. The resulting preprocessed tokens are divided into training and test sets based on specific criteria detailed [here](https://github.com/zbmed-semtec/relish-preprocessing?tab=readme-ov-file#splitting-the-data). These splits are then saved as two separate .npy files.

Additionally, the ground truth relevance assessments are used to evaluate the accuracy of the doc-2-doc recommendations, ensuring that the method's results align with expert judgments.


## 🛠️Pipeline

The following section outlines the process of generating document-level embeddings out of word-level embeddings for each PMID of the RELISH corpus through hyperparameter optimization, computing the cosine similarity scores and evaluating the given similarity results with the relevance matrix.

## 🧠⚙️ Train and Optimize Word2vec models
We create and train Word2vec models with customizable hyperparameters to comprehend the connections between documents and words in a high-dimensional vector space. We aim to optimize these hyperparameters to establish the most effective relationship between cosine similarity and document relevance.

To accomplish this we begin by splitting the dataset into a training set and a testing set. The training set is then used to train the Word2vec model, where we explore various hyperparameters to optimize its performance. This optimization process is crucial for enhancing the model's ability to capture meaningful relationships between cosine similarity and document relevance. For each set of hyperparameters, a Word2vec model is trained on the training split.

Following this, we evaluate the model's performance on the testing set using Precision@5 as our evaluation metric.

##### Parameters

+ **sg:** {1,0} Refers to the training algorithm. If sg=1, skip-gram is used otherwise, continuous bag of words is used.
+ **vector_size:** It represents the dimensions of the generated embeddings, with options of 200, 300 and 400 in our case.
+ **window:** Represents the maximum distance between the current and predicted word, with values fof 5,6 and 7 in our case.
+ **epochs:** Refers to the number of iterations over the training dataseta and is set at 15 in this context.
+ **min_count:** It is the minimum number of appearances a word must have to not be ignored by the algorithm and is configured at 1, 2 and 3 in our case.

## 📐🔄Calculate Cosine Similarity

Following hyperparameter optimization where the best model gets saved, embeddings are generated for the test dataset using this trained model. Subsequently, cosine similarity is calculated for the test dataset embeddings, providing a measure of similarity between pairs of documents based on their learned representations. This enables the generation of a 4-column matrix [ PMID1 | PMID2 | Relevance | Cosine similarity ] containing cosine similarity scores for existing pairs of PMIDs within our corpus. For a more detailed explanation of the process, please refer to this [documentation](https://github.com/zbmed-semtec/medline-preprocessing/tree/main/code/Cosine_Similarity).

## 📈📋 Evaluation

The effectiveness of the embeddings in capturing document-to-document similarity is assessed using two metrics: Precision@N and nDCG@N.

### 🎯Precision@N

Precision@N measures the precision of retrieved documents at various cutoff points (N).We generate a Precision@N matrix for existing pairs of documents within the RELISH corpus, based on the original RELISH JSON file. The [code](code/precision.py) determines the number of true positives within the top N pairs and computes Precision@N scores. The result is a Precision@N matrix with values at different cutoff points, including average scores. For detailed insights into the algorithm, please refer to this [documentation](https://github.com/zbmed-semtec/medline-preprocessing/tree/main/code/Precision%40N_existing_pairs).

### 📊 nDCG@N

Another metric used is the nDCG@N (normalized Discounted Cumulative Gain). This ranking metric assesses document retrieval quality by considering both relevance and document ranking. It operates by using a TSV file containing relevance and cosine similarity scores, involving the computation of DCG@N and iDCG@N scores. The result is an nDCG@N matrix for various cutoff values (N) and each PMID in the corpus, with detailed information available in the [documentation](https://github.com/zbmed-semtec/medline-preprocessing/tree/main/code/Evaluation).


## 🧑‍💻🧩 Code Implementation

+ The [`main.py`](code/main.py) serves as a comprehensive wrapper function, supporting the model generation, training, embedding generation, cosine similarity matrix calculation, precision calculation and gain calculation in one pipeline. Individual functions for each task are provided in the other scripts.

+ [`optunaTuningUnix.py`](code/optunaTuningUnix.py) / [`optunaTuningWindows.py`](code/optunaTuningWindows.py) : The code utilizes Optuna for hyperparameter optimization of Word2vec model. It suggests hyperparameters for Word2vec, trains models, evaluates precision@5, and selects the best trial. The optimization process iterates over several trials, updating progress with a progress bar. The scripts are designed to run the pipeline on either Unix or Windows systems.

+ [`train.py`](code/train.py): This script trains a Word2vec model using specified hyperparameters, saves the model if specified, generates embeddings for test data, computes cosine similarity scores, and saves them to a file. It logs progress to a file specified by log_file.

+ [`utilities.py`](code/utilities.py): This script includes functions for parsing and reading input tokens, creation and training of Word2vec models, generation of embeddings, centroid aggregation of word embeddings to generate document embeddings, calculation of cosine similarity, generation of similarity matrix.

+ [`precision.py`](code/precision.py): This script reads a TSV file containing cosine similarity pairs, calculates precision scores at various values of n for each PMID, and writes the results along with average precision scores to a new TSV file.

+ [`calculate_gain.py`](code/calculate_gain.py): This script calculates normalized discounted cumulative gain (nDCG) scores for relevance assessment based on cosine similarity values, sorts data accordingly, and writes results including average nDCG scores to a TSV file. It utilizes the cosine similarity matrix provided and performs operations per PMID.


## 🚀 Getting Started

To get started with this project, follow these steps:

### Step 1: Clone the Repository
First, clone the repository to your local machine using the following command:

###### Using HTTP:

```
git clone https://github.com/zbmed-semtec/word2doc2vec-doc-relevance-training.git
```

###### Using SSH:
Ensure you have set up SSH keys in your GitHub account.

```
git clone git@github.com:zbmed-semtec/word2doc2vec-doc-relevance-training.git
```

### Step 2: Create a virtual environment and install dependencies

To create a virtual environment within your repository, run the following command:

```
python3 -m venv .venv 
source .venv/bin/activate   # On Windows, use '.venv\Scripts\activate' 
```

To confirm if the virtual environment is activated and check the location of yourPython interpreter, run the following command:

```
which python    # On Windows command prompt, use 'where python'
                # On Windows PowerShell, use 'Get-Command python'
```
The code is stable with python 3.9 and higher. The required python packages are listed in the requirements.txt file. To install the required packages, run the following command:

```
pip install -r requirements.txt
```

To deactivate the virtual environment after running the project, run the following command:

```
deactivate
```

### Step 3: Dataset


- Use the [Download_Dataset.sh](./Download_Dataset.sh) script to download the Split Dataset by running the following commands:

```
chmod +777 Download_Dataset.sh
./Download_Dataset.sh
```
This script makes sure that the necessary folders are created and the files are downloaded in the corresponding folders.

**OR**


- You could also download the dataset from this link: [Split_Dataset](https://drive.google.com/drive/folders/1Bq_U5207utn7tvSt_HLVdOdYR5QW7MMN). Please make sure to keep the data in the below specified format.

```
📦 /word2doc2vec-doc-relevance-training
└─ data
   └─ Split_Dataset
      ├─ Data
      │  ├─ train.npy
      │  ├─ test.npy
      └─ Ground_truth
         ├─ train.tsv
         └─ test.tsv

```


### Step 4: Optimization Pipeline

This pipeline aims to optimize hyperparameters for a Word2vec model using Optuna, train the model with the optimal parameters, and evaluate its performance using precision at N (Precision@N) and normalized discounted cumulative gain (NDCG) metrics.

Steps:
+ Hyperparameter Optimization: Utilizes Optuna to search for the best hyperparameters for the fastText model.
+ Model Training: Trains the fastText model with the optimal hyperparameters using 80% of the training split data.
+ Embedding Generation: Generates embeddings for the remaining 20% of the test split data using the trained model.
+ Cosine Similarity Computation: Calculates cosine similarities for the generated embeddings.
+ Precision@N Calculation: Computes Precision@N scores, a measure of the relevance of retrieved documents, for the obtained cosine similarities.
+ NDCG Score Calculation: Computes normalized discounted cumulative gain (NDCG) scores, which assesses the quality of ranked search results based on relevance assessments.

In order to start the pipeline execution use this script, and run the following command:

 ``` 
python3 code/main.py [-i INPUT TRAIN FILE] [-t TEST_FILE] [-g GROUND_TRUTH_FILE] [-c NO_OF CLASSES] [-win WINDOWS/LINUX]
 ``` 

 You must pass the following four arguments:

+ -i/ --input : File path to the RELISH Train split dataset (.npy file format).
+ -t/ --test : File path to the RELISH Test split dataset (.npy file format).
+ -g/ --ground_truth : File path for the Test split ground truth (.tsv file format).
+ -c/ --classes : No. of classes to perform optimization on (Integer 2 or 3/ Default value is 3).
+ -win/ --windows : 1 - if using Windows systems; 0 - if using Unix-like systems (including Ubuntu)

To run this script, please execute the following command:

 ``` 
python3 code/main.py -i data/Split_Dataset/Data/train.npy -t data/Split_Dataset/Data/test.npy -gt data/Split_Dataset/Ground_truth/test.tsv -c 3 -win 0
 ``` 

Precision@N and NDCG scores are saved as TSV files in the following folder path: `\output_2\evaluation\`  for 2 class distribution and `\output_3\evaulation\` for 3 class distribution for further analysis and reporting.

Make sure to run the model training twice for both the class distributions by changing the value of the -c/ --classes flag to 2 and 3.
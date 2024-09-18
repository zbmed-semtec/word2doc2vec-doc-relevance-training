import os
import time
import argparse
import logging
import subprocess
import utilities as utilities
import gensim.downloader as api
import gensim.models as model
from gensim.models import Word2Vec

def run_pretrained(args, model_directory):

    # 1) Load the pre-trained fastText model
    try:
        model = api.load('word2vec-google-news-300')
        logging.info("Pre-trained Word2Vec model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load the pre-trained model: {e}")
        return

    # 2) Set the test data to be used based on tuning parameter
    dataset_type = "Test"
    data_file = args.test
    ground_truth = args.ground_truth

    # 3) Load the data from npy file
    pmids, docs = utilities.process_data_from_npy(data_file)
    logging.info(f"Retrieved RELISH Cleaned {dataset_type} Data")

    # 4) Generate the embeddings: pd.DataFrame for loaded docs
    embeddings_df, null_vector_count = utilities.generate_embeddings(model, pmids, docs, args.use_pretrained)
    logging.info(f"RELISH {dataset_type} Embeddings Pickle File Generated.")
    logging.info(f"Number of null vector documents in test data: {null_vector_count}")


    # 5) Generate the cosine similarity matrix: pd.DataFrame for the generated embeddings
    similarity_df = utilities.get_similarity_scores(ground_truth, embeddings_df)
    logging.info(f"RELISH {dataset_type} Cosine Similarity Matrix Generated.")

    embeddings_file = f"output_{args.classes}/embeddings/best_embeddings_{args.classes}.pkl"
    similarity_file = f"output_{args.classes}/evaluation/best_cosine_similarity_{args.classes}.tsv"
    
    utilities.save_embeddings_to_pickle(embeddings_df, embeddings_file)
    utilities.save_similarity_to_tsv(similarity_df, similarity_file)
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

    # 1) Load the pre-trained Word2vec model
    try:
        model = api.load('word2vec-google-news-300')
        logging.info("Pre-trained Word2Vec model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load the pre-trained model: {e}")
        return

    # 2) Load the data from npy file
    test_pmids, test_docs = utilities.process_data_from_npy(args.test)
    logging.info(f"Retrieved RELISH Cleaned Test Data")

    # 3) Generate the embeddings: pd.DataFrame for loaded docs
    test_embeddings_df, null_vector_count = utilities.generate_embeddings(model, test_pmids, test_docs, args.use_pretrained)
    logging.info(f"RELISH Test Embeddings Pickle File Generated.")
    logging.info(f"Number of null vector documents in test data: {null_vector_count}")

    # 4) Generate the cosine similarity matrix: pd.DataFrame for the generated embeddings
    test_similarity_df = utilities.get_similarity_scores(args.test_ground_truth, test_embeddings_df)
    logging.info(f"RELISH Test Cosine Similarity Matrix Generated.")

    test_embeddings_file = f"output_{args.classes}/embeddings/best_embeddings_{args.classes}.pkl"
    test_similarity_file = f"output_{args.classes}/evaluation/best_cosine_similarity_{args.classes}.tsv"
    
    utilities.save_embeddings_to_pickle(test_embeddings_df, test_embeddings_file)
    utilities.save_similarity_to_tsv(test_similarity_df, test_similarity_file)

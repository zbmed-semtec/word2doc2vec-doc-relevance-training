import tqdm
import gensim
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from typing import Union, List
import logging

log_file = 'output.log'
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s')

# Retrieves cleaned data from RELISH and TREC npy files


def process_data_from_npy(file_path_in: str = None) -> Union[List[str], List[List[str]], List[List[str]], List[List[str]]]:
    """
    Retrieves cleaned data from RELISH and TREC npy files, separating each column 
    into their own respective list.

    Parameters
    ----------
    filepathIn: str
            The filepath of the RELISH or TREC input npy file.
    Returns
    -------
    pmids: List[str]
            A list of all pubmed ids in the corpus.
    titles: List[List[str]]
            A list of lists where each sub-list contains the words 
            in the cleaned/processed title.
    abstracts: List[List[str]]
            A list of lists where each sub-list contains the words 
            in the cleaned/processed abstract.
    docs: List[List[str]]
            A list of lists where each sub-list contains the words 
            in the cleaned/processed document (title + abstract).
    """
    doc = np.load(file_path_in, allow_pickle=True)

    pmids = []
    docs = []

    for line in doc:
        pmids.append(line[0])
        if type(line[1]) == str:
            title_content = line[1].strip('][').split(', ')
            title = ' '.join(title_content).replace("\'", "")
            title_tokens = title.split(" ")
        else:
            title_tokens = line[1]

        if type(line[2]) == str:
            abstract_content = line[2].strip('][').split(', ')
            abstract = ' '.join(abstract_content).replace("\'", "")
            abstract_tokens = abstract.split(" ")
        else:
            abstract_tokens = line[2]

        docs.append(title_tokens + abstract_tokens)

    return (pmids, docs)


def generate_Word2Vec_model(article_doc: list, pmids: list, params: list, filepath_out: str):
    '''
    Generates a word2vec model from all RELISH sentences using gensim and saves it as a .model file.

    Parameters
    ----------
    article_doc: list of list of str
        A two dimensional list of all tokenized article documents (title + abstract).
    pmids: list of str
        A list of all appearing pubmed ids in the input dataset.
    params: dict
        A dictionary of the hyperparameters for the model.
    filepath_out: str
        The filepath for the resulting word2vec model file.
    '''
    from gensim.models import Word2Vec
    sentence_list = []
    for index in range(len(pmids)):
        sentence_list.append(article_doc[index])
    params['sentences'] = sentence_list
    wv_model = None
    wv_model = Word2Vec(**params)
    wv_model.save(filepath_out)


def generate_document_embeddings(pmids: str, article_doc: list, params: list):
    '''
    Generates document embeddings from a titles and abstracts in a given paper using word2vec and calculating the cenroids of all given word embeddings.
    If no gensim model is given, the 'glove-wiki-gigaword-200' gensim model is used.

    Parameters
    ----------
    pmids: list of str
        The list of all pmids which are processed.
    article_doc: list of list of str
        A two dimensional list of all tokenized article documents (title + abstract).
    gensim_model_path: str (optional)
        The filepath of the custom gensimModel.
    '''
    import gensim.downloader as api
    import gensim.models as model
    import time
    import os

    st = time.time()
    filepath_out = "data/model"

    generate_Word2Vec_model(article_doc, pmids, params, filepath_out)

    word_vectors = model.Word2Vec.load(filepath_out)
    missing_words = 0
    iteration = 0
    document_embeddings = []
    for iteration in range(len(pmids)):
        # Retrieve word embeddings.
        embedding_list = []
        for word in article_doc[iteration]:
            try:
                embedding_list.append(word_vectors.wv[word])
            except:
                missing_words += 1

        # Generate document embeddings from word embeddings using word-vector centroids.
        if len(embedding_list) == 0:
            # This can be caused by a high min-count parameter or missing vocabulary when using a pretrained model
            document_embeddings.append([])
            continue
        document = [0.0] * word_vectors.vector_size

        for dim in range(word_vectors.vector_size):
            for word_embeddings in embedding_list:
                document[dim] += word_embeddings[dim]
            document[dim] = document[dim] / len(embedding_list)
        document_embeddings.append(document)

    et = time.time()

    # get the execution time
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')

    import pandas as pd
    df = pd.DataFrame(list(zip((pmids), document_embeddings)),
                      columns=['PID', 'Embedding'])
    df = df.sort_values('pmids')
    return df
    # os.makedirs(f"{directory_out}/{param_iteration}", exist_ok=True)
    # df.to_pickle(f'{directory_out}/{param_iteration}/embeddings.pkl')


def saveWord2Doc2VecModel(df, output_file):
    df.to_pickle(f'{output_file}')


# def createDoc2VecModel(pmids: List[str], docs: List[List[str]], params: dict) -> Doc2Vec:
#     """
#     Create and train the Doc2Vec model using Gensim for the documents
#     in the corpus.

#     Parameters
#     ----------
#     pmids: List[str]
#             A list of all pubmed ids in the corpus.
#     docs: List[List[str]]
#             A list of lists where each sub-list contains the words
#             in the cleaned/processed document (title + abstract).
#     params: dict
#             Dictionary containing the parameters for the Doc2Vec model.
#     Returns
#     -------
#     model: Doc2Vec
#             Doc2Vec model.
#     """
#     tagged_data = [TaggedDocument(words=_d, tags=[str(pmids[i])])
#                    for i, _d in enumerate(docs)]

#     # model = Doc2Vec(vector_size=200, window=5, min_count=1, epochs=5)
#     model = Doc2Vec(**params)
#     model.build_vocab(tagged_data)
#     model.train(tagged_data, total_examples=model.corpus_count,
#                 epochs=model.epochs)

#     return model

# # Save the Doc2Vec Model


# def saveDoc2VecModel(model: Doc2Vec, output_file: str) -> None:
#     """
#     Saves the Doc2Vec model.

#     Parameters
#     ----------
#     model: Doc2Vec
#             Doc2Vec model.
#     output_file: str
#             File path of the Doc2Vec model generated.
#     """
#     model.save(output_file)


def calculate_cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)


def get_similarity_scores(input_relevance_matrix, embeddings, output_matrix_name):
    # Read Embeddings
    embeddings_df = pd.read_pickle(embeddings)

    logging.info("Embeddings DataFrame Loaded")

    # Read Relevance matrix
    column_names = ["PID1", "PID2", "Value"]
    relevance_matrix_df = pd.read_csv(
        input_relevance_matrix, sep="\t", names=column_names, skiprows=1)

    # Adds empty columns to the file to store similarity scores
    relevance_matrix_df["Cosine Similarity"] = ""

    embeddings_dict = {int(pmid): embedding for pmid, embedding in zip(
        embeddings_df['PID'], embeddings_df['Embedding'])}

    # Create a list of ref and assessed PMID pairs
    pmid_pairs = list(
        zip(relevance_matrix_df["PID1"], relevance_matrix_df["PID2"]))

    for ref_pmid, assessed_pmid in tqdm.tqdm(pmid_pairs, total=len(pmid_pairs), desc="Calculating Similarities"):
        try:
            ref_pmid_vector = embeddings_dict[ref_pmid]
            assessed_pmid_vector = embeddings_dict[assessed_pmid]
            if ref_pmid_vector is not None and assessed_pmid_vector is not None:
                cosine_similarity = round(calculate_cosine_similarity(
                    ref_pmid_vector, assessed_pmid_vector), 4)
                relevance_matrix_df.loc[(relevance_matrix_df['PID1'] == ref_pmid) & (
                    relevance_matrix_df['PID2'] == assessed_pmid), 'Cosine Similarity'] = cosine_similarity
            else:
                logging.info(
                    f"One of the vectors is None for ({ref_pmid}, {assessed_pmid})")
        except KeyError as e:
            logging.info(
                f"\nKeyError: {e}, ref_pmid: {ref_pmid}, assessed_pmid: {assessed_pmid}")
            break

    print('Added similarity scores')

    # Saves the updated matrix
    relevance_matrix_df.to_csv(output_matrix_name, index=False, sep="\t")
    logging.info('Saved matrix')


def generate_embeddings(modeldf, pmids, output_file):
    embeddings_list = []
    for i in range(len(pmids)):
        print(modeldf['PID'])
    save_embeddings_to_pickle(pmids, embeddings_list, output_file)


def save_embeddings_to_pickle(pmids, embeddings_list, output_file):
    data = {"PID": pmids, "Embedding": embeddings_list}
    df = pd.DataFrame(data)
    df.to_pickle(output_file)
    print(f"Embeddings saved to {output_file}")

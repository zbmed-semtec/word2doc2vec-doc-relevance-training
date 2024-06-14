import tqdm
import gensim
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from gensim.models import Word2Vec
from typing import Union, List

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
    article_docs = []

    for line in range(len(doc)):
        pmids.append(int(doc[line][0]))

        # Check if the element is a NumPy array before using tolist
        if isinstance(doc[line][1], np.ndarray):
            article_docs.append(doc[line][1].tolist())
        else:
            article_docs.append(doc[line][1])

        # Check if the element is a NumPy array before using tolist
        if isinstance(doc[line][2], np.ndarray):
            article_docs[line].extend(doc[line][2].tolist())
        else:
            article_docs[line].extend(doc[line][2])
    return (pmids, article_docs)

# def createWord2VecModel(docs: List[List[str]], params: dict) -> Word2Vec:
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
#     # tagged_data = [TaggedDocument(words=_d, tags=[str(pmids[i])])
#     #                for i, _d in enumerate(docs)]
#     # import itertools
#     # corpus_iterable = list(itertools.chain.from_iterable(docs))
#     sentence_list = []
#     for doc in docs:
#         sentence_list.append(doc)
#     params['sentences'] = sentence_list
#     model = Word2Vec(**params)
#     # model.build_vocab(corpus_iterable)
#     # model.train(corpus_iterable, total_examples=model.corpus_count,
#     #             epochs=model.epochs)

#     return model


def createWord2VecModel(docs: List[List[str]], params: dict) -> Word2Vec:
    """
    Create and train the Word2Vec model using Gensim for the words of documents 
    in the corpus.

    Parameters
    ----------
    docs: List[List[str]]
            A list of lists where each sub-list contains the words 
            in the cleaned/processed document (title + abstract).
    params: dict
            Dictionary containing the parameters for the Word2Vec model.
    Returns
    -------
    model: Word2Vec
            Word2Vec model.
    """
    # tagged_data = [TaggedDocument(words=_d, tags=[str(pmids[i])])
    #                for i, _d in enumerate(docs)]
    import itertools
    corpus_iterable = list(itertools.chain.from_iterable(docs))
    sentence_list = []
    for doc in docs:
        sentence_list.append(doc)
    # params['sentences'] = sentence_list
    model = Word2Vec(**params)
    model.build_vocab(corpus_iterable)
    model.train([corpus_iterable], total_examples=model.corpus_count,
                epochs=model.epochs)
    model.save("word2vec_RELISH")
    print(len(corpus_iterable))
    print(f"Model length: {len(model.wv)}")
    return model


def saveWord2VecModel(model: Word2Vec, output_file: str) -> None:
    """
    Saves the Word2Vec model.

    Parameters
    ----------
    model: Word2Vec
            Word2Vec model.
    output_file: str
            File path of the Word2Vec model generated.
    """
    model.save(output_file)


def loadWord2VecModel(model_path: str) -> None:
    """
    Loads the saved Word2Vec model.

    Parameters
    ----------
    model_path: str
            Path of the Word2Vec model.

    Return
    ----------
    model: Word2Vec
            Word2Vec model.
    """
    model = gensim.models.Word2Vec.load(model_path)
    return model


def calculate_cosine_similarity(vector_1: np.ndarray, vector_2: np.ndarray) -> float:
    """
    Calculate the cosine similarity between two vectors.

    This function computes the cosine similarity, which is defined as 1 minus the cosine distance 
    between two vectors. Cosine similarity is a measure of similarity between two non-zero vectors
    of an inner product space that measures the cosine of the angle between them.

    Parameters:
    ----------
    vector_1 : np.ndarray
        A numpy array representing the first vector.
    vector_2 : np.ndarray
        A numpy array representing the second vector.

    Returns:
    -------
    float
        The cosine similarity between vector_1 and vector_2.
    """
    return 1 - cosine(vector_1, vector_2)


def get_similarity_scores(input_relevance_matrix: str, embeddings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate cosine similarity scores for pairs of PubMed IDs based on their embeddings and update a DataFrame with these scores.

    Parameters:
    ----------
    input_relevance_matrix : str
        File path to the TSV file containing pairs of PubMed IDs and a relevance value.
    embeddings_df : pd.DataFrame
        DataFrame containing PubMed IDs and their corresponding document embeddings.

    Returns:
    -------
    relevance_matrix_df : pd.DataFrame
        Updated DataFrame with cosine similarity scores added for each pair.
    """

    # 1) Read Relevance matrix
    column_names = ["PID1", "PID2", "Value"]
    relevance_matrix_df = pd.read_csv(
        input_relevance_matrix, sep="\t", names=column_names, skiprows=1)

    # 2) Adds empty columns to the file to store similarity scores
    relevance_matrix_df["Cosine Similarity"] = ""

    # 3) Create a dictionary to store embeddings
    embeddings_dict = {int(pmid): embedding for pmid, embedding in zip(
        embeddings_df['PID'], embeddings_df['Embedding'])}

    # 4) Create a list of reference and assessed PMID pairs
    pmid_pairs = list(
        zip(relevance_matrix_df["PID1"], relevance_matrix_df["PID2"]))

    # 5) Calculate the cosine similarities between the document embeddings and update the relevance matrix dataframe
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
                # if ref_pmid_vector == None:
                #     print(f"Missing ref_pmid: {ref_pmid}")
                # if assessed_pmid_vector == None:
                #     print(f"Missing assessed_pmid: {assessed_pmid}")
                continue
        except KeyError as e:
            print(
                f"\nKeyError: {e}, ref_pmid: {ref_pmid}, assessed_pmid: {assessed_pmid}")
            break

    return relevance_matrix_df


def save_similarity_to_tsv(df: pd.DataFrame, output_file: str) -> None:
    """
    Save the DataFrame containing similarity scores to a TSV file.

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame to be saved, containing similarity scores among other data.
    output_file : str
        The file path where the DataFrame will be saved as a TSV.
    """
    df.to_csv(output_file, index=False, sep="\t")


def generate_embeddings(pmids: List[str], docs: List[List[str]]) -> pd.DataFrame:
    """
    Generate embeddings for a list of documents using a trained Word2Vec model.

    Parameters:
    ----------
    model : Word2Vec
        The trained Word2Vec model used for generating embeddings.
    pmids : List[str]
        List of PubMed IDs corresponding to the documents.
    docs : List[List[str]]
        List of documents, where each document is represented as a list of words.

    Returns:
    -------
    embeddings_df : pd.DataFrame
        DataFrame containing PubMed IDs and their corresponding embeddings.
    """
    document_embeddings = []
    all_words = 0
    missing_words = 0
    # Add new words to word2vec model
    new_words = []
    model = Word2Vec.load("word2vec_RELISH")
    for doc in docs:
        for word in doc:
            if word not in model.wv:
                new_words.append(word)
    # min_count needs to be set to 1, otherwise some docs won't have a single word recognized.
    model.min_count = 1
    model.build_vocab([new_words], update=True)
    model.train(new_words, total_examples=model.corpus_count + len(new_words),
                epochs=model.epochs)
    print(f"Model length: {len(model.wv)}")
    for doc in docs:
        # Infer vector for each document
        # vector = model.infer_vector(doc)
        # document_embeddings.append(vector)

        # Retrieve word embeddings.
        embedding_list = []
        missing_words_list = []
        for word in doc:
            if (word in model.wv):
                embedding_list.append(model.wv[word])
                all_words += 1
            else:
                missing_words += 1
                missing_words_list.append(word)
                print(f"Missing word is in new_words: {word in new_words}")

        # Generate document embeddings from word embeddings using word-vector centroids.
        if len(embedding_list) == 0:
            # This can be caused by a high min-count parameter or missing vocabulary when using a pretrained model
            document_embeddings.append(None)
            continue
        vector = [0.0] * model.vector_size

        for dim in range(model.vector_size):
            for word_embeddings in embedding_list:
                vector[dim] += word_embeddings[dim]
            vector[dim] = vector[dim] / len(embedding_list)
        document_embeddings.append(vector)

    data = {"PID": pmids, "Embedding": document_embeddings}
    embeddings_df = pd.DataFrame(data)
    embeddings_df = embeddings_df.sort_values("PID")
    return embeddings_df


def save_embeddings_to_pickle(df: pd.DataFrame, output_file: str) -> None:
    """
    Save the DataFrame containing document embeddings to a pickle file.

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing embeddings to be saved.
    output_file : str
        The file path where the DataFrame will be saved in pickle format.
    """
    df.to_pickle(output_file)

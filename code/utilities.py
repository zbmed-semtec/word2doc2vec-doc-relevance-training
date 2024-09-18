import tqdm
import gensim
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from gensim.models import Word2Vec
from typing import Union, List

def createWord2VecModel(docs: List[List[str]], params: dict):
    '''
    Create and train the Word2Vec model using Gensim for the documents in the corpus.

    Parameters
    ----------
    pmids: List[str]
            A list of all pubmed ids in the corpus.
    docs: List[List[str]]
            A list of lists where each sub-list contains the words 
            in the cleaned/processed document (title + abstract).
    params: dict
            Dictionary containing the parameters for the Doc2Vec model.
    '''
    # Each document is treated as a separate sentence
    model = Word2Vec(sentences=docs, **params)
    # Train the model using the sentences
    model.train(docs, total_examples=model.corpus_count, epochs=model.epochs)
    # Save the model to a file
    model.save("word2vec_RELISH")
    return model

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

def saveWord2Doc2VecModel(model: Word2Vec, output_file: str) -> None:
    """
    Saves the Word2Doc2Vec model.

    Parameters
    ----------
    model: Word2Vec
            Word2Vec model.
    output_file: str
            File path of the Word2Vec model generated.
    """
    model.save(output_file)

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
    column_names = ["PMID1", "PMID2", "Value"]
    relevance_matrix_df = pd.read_csv(
        input_relevance_matrix, sep="\t", names=column_names, skiprows=1)

    # 2) Adds empty columns to the file to store similarity scores
    relevance_matrix_df["Cosine Similarity"] = ""

    # 3) Create a dictionary to store embeddings
    embeddings_dict = {int(pmid): embedding for pmid, embedding in zip(
        embeddings_df['PMID'], embeddings_df['Embedding'])}

    # 4) Create a list of reference and assessed PMID pairs
    pmid_pairs = list(
        zip(relevance_matrix_df["PMID1"], relevance_matrix_df["PMID2"]))

    # 5) Calculate the cosine similarities between the document embeddings and update the relevance matrix dataframe
    for ref_pmid, assessed_pmid in tqdm.tqdm(pmid_pairs, total=len(pmid_pairs), desc="Calculating Similarities"):
        try:
            ref_pmid_vector = embeddings_dict[ref_pmid]
            assessed_pmid_vector = embeddings_dict[assessed_pmid]
            if len(ref_pmid_vector) > 0 and len(assessed_pmid_vector) > 0:
                cosine_similarity = round(calculate_cosine_similarity(
                    ref_pmid_vector, assessed_pmid_vector), 4)
                relevance_matrix_df.loc[(relevance_matrix_df['PMID1'] == ref_pmid) & (
                    relevance_matrix_df['PMID2'] == assessed_pmid), 'Cosine Similarity'] = cosine_similarity
            else:
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

def generate_embeddings(model: Word2Vec, pmids: str, article_doc: list, pre_trained: int) -> pd.DataFrame:
    '''
    Generates document embeddings from titles and abstracts in a given paper using Word2Vec and calculating the centroids 
    of all given word embeddings.
    
    Parameters
    ----------
    model: Word2Vec
        Word2Vec model.
    pmids: list of str
        The list of all PMIDs which are processed.
    article_doc: list of list of str
        A two-dimensional list of all tokenized article documents (title + abstract).
    pre_trained: int
        Whether to use a pre-trained model or not.
    Returns
    -------
    embeddings_df : pd.DataFrame
        DataFrame containing PubMed IDs and their corresponding embeddings.
    null_vector_count : int
        The count of documents that resulted in a null vector.
    '''

    data_dict = {}
    missing_words = 0
    word_count = 0
    iteration = 0
    document_embeddings = []
    null_vector_count = 0  # Initialize counter for null vector documents

    for iteration in range(len(pmids)):
        missing_words = 0
        # Retrieve word embeddings.
        embedding_list = []
        if pre_trained==1:
            for word in article_doc[iteration]:
                word_count += 1
                try:
                    embedding_list.append(model[word])
                except:
                    missing_words += 1
        else:
            for word in article_doc[iteration]:
                word_count += 1
                try:
                    embedding_list.append(model.wv[word])
                except:
                    missing_words += 1
        if missing_words == word_count:
            print(f"OOV words for {pmids[iteration]}: {missing_words} from a total of {word_count} words")
        word_count = 0

        # Generate document embeddings from word embeddings using word-vector centroids.
        if len(embedding_list) == 0:
            document_embeddings.append([])
            null_vector_count += 1  # Increment the counter when null vector is encountered
            continue

        document = [0.0] * model.vector_size
        for dim in range(model.vector_size):
            for word_embeddings in embedding_list:
                document[dim] += word_embeddings[dim]
            document[dim] = document[dim] / len(embedding_list)
        document_embeddings.append(document)

    data = {"PMID": pmids, "Embedding": document_embeddings}
    embeddings_df = pd.DataFrame(data)
    embeddings_df = embeddings_df.sort_values("PMID")
    return embeddings_df, null_vector_count

def save_embeddings_to_pickle(embeddings_df, output_file):
    embeddings_df.to_pickle(output_file)
    print(f"Embeddings saved to {output_file}")

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



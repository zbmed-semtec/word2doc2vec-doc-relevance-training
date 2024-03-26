import argparse
import pandas as pd
import numpy as np
from typing import List, Tuple

def read_file(tsv_file: str) -> Tuple[List, pd.DataFrame]:
    """
    Reads the input 4-column cosine similarity existing pairs TSV file in a pandas dataframe, generates all the unique PMIDs
    and returns the dataframe.
    Parameters
    ----------
    tsv_file : str
        File path to the 4-column cosine similarity existing pairs TSV file.
    Returns
    -------
    ref_pmids : list
        List of all unique PMIDs.
    data : pd.Dataframe
        Pandas Dataframe cosisting of 4 columns: PMID1, PMID2, Relevance, Cosine similarity.
    """
    colnames = ["PID1", "PID2", "Value", "Cosine Similarity"]
    data = pd.read_csv(tsv_file, sep='\t', header=0, names=colnames)
    ref_pmids = data["PID1"].unique()
    return ref_pmids, data


def sort_collection(pmid: str, data: pd.DataFrame) -> pd.DataFrame:
    """
    Sorts the input dataframe for the given PMID based on the cosine similarity values in the descending order.
    Parameters
    ----------
    pmid : str
        PMID for which the collection needs to be sorted.
    data : pd.Datafarme
        Pandas Dataframe cosisting of 4 columns: PMID1, PMID2, Relevance, Cosine similarity.
    Returns
    -------
    sorted_collection : pd.Dataframe
        Sorted Pandas Dataframe based on the given PMID .
    """
    collection = data[data['PID1'] == pmid]
    sorted_collection = collection.sort_values(['PID1', 'Cosine Similarity'],
                                               ascending=[True, False], ignore_index=True)
    return sorted_collection


def calculate_precision(sorted_collection: pd.DataFrame, n: int, classes: int) -> float:
    """
    Calculates the precision score for the input sorted_collection at given n value.
    Parameters
    ----------
    sorted_collection : pd.Dataframe
        Sorted Pandas Dataframe based on the given PMID .
    n : int
        Value of n at which precision is to be calculated.
    classes: int
        Number of classes to take into consideration
    Returns
    -------
    precision_n : float
        Value of Precision@n.
    """
    top_n = sorted_collection[:n]
    if int(classes) == 2:
        true_positives_n = len(top_n[(top_n["Value"] == 2) | (top_n["Value"] == 1)]) # two classes
    else:
        true_positives_n = len(top_n[top_n["Value"] == 2])  # three classes
    precision_n = round(true_positives_n/n, 4)
    return precision_n

def generate_vector(ref_pmids: list, data: pd.DataFrame, classes: int) -> np.array:
    """
    Wrapper function to generate the precision vector at the values of n=5 for every unique PMID in the input data.
    Parameters
    ----------
    ref_pmids : list
        List of all unique PMIDs.
    data : pd.Dataframe
        Pandas Dataframe cosisting of 4 columns: PMID1, PMID2, Relevance, Cosine similarity.
    classes: int
        no. of classes to consider
    Returns
    -------
    precision_vector : np.array
        Generated precision vector.
    """
    value_of_n = 5
    ref_pmids_filtered = [pmid for pmid in ref_pmids if len(data[data['PID1'] == pmid]) >= value_of_n]
    precision_vector = np.empty(shape=(len(ref_pmids_filtered), 1))
    for pmid_index, pmid in enumerate(ref_pmids_filtered):
        sorted_collection = sort_collection(pmid, data)
        precision_n = calculate_precision(sorted_collection, value_of_n, classes)
        precision_vector[pmid_index][0] = precision_n
        
    return precision_vector

def generate_matrix(ref_pmids: list, data: pd.DataFrame, classes: int) -> np.array:
    """
    Wrapper function to generate the precision vector at the given values of n for every unique PMID in the input data.
    Parameters
    ----------
    ref_pmids : list
        List of all unique PMIDs.
    data : pd.Dataframe
        Pandas Dataframe cosisting of 4 columns: PMID1, PMID2, Relevance, Cosine similarity.
    Returns
    -------
    precision_matrix : np.array
        Generated precision matrix.
    """
    value_of_n = [5, 10, 15, 20]
    ref_pmids_filtered = [pmid for pmid in ref_pmids if len(data[data['PID1'] == pmid]) >= max(value_of_n)]
    precision_matrix = np.empty(shape=(len(ref_pmids_filtered), len(value_of_n)))
    for pmid_index, pmid in enumerate(ref_pmids_filtered):
        sorted_collection = sort_collection(pmid, data)
        for index, n in enumerate(value_of_n):
            precision_n = calculate_precision(sorted_collection, n, classes)
            precision_matrix[pmid_index][index] = precision_n
    return precision_matrix

def write_to_tsv(ref_pmids: list, precision_matrix: np.array, output_filepath: str, data: pd.DataFrame):
    """
    Write the generated precision matrix to a TSV file and computes the avergae of the precision@n scores.
    Parameters
    ----------
    ref_pmids: list
        List of all unique PMIDs.
    precision_matrix : np.array
        Generated precision matrix.
    output_filepath : str
        File path to save the TSV file.
    data : pd.Dataframe
        Pandas Dataframe cosisting of 4 columns: PMID1, PMID2, Relevance, Cosine similarity.
    """
    # Filter out PMIDs with less than 20 instances
    ref_pmids_filtered = [pmid for pmid in ref_pmids if len(data[data['PID1'] == pmid]) >= 20]
    matrix = pd.DataFrame(precision_matrix, columns=['P@5', 'P@10', 'P@15', 'P@20'])

    matrix.insert(0, 'PIDs', ref_pmids_filtered)
    # Calculate and append average of each precision score
    average_values = ['Average'] + list(matrix[['P@5', 'P@10', 'P@15', 'P@20']]
                                        .mean(axis=0).round(4))
    matrix.loc[len(matrix.index)] = average_values
    pd.DataFrame(matrix).to_csv(output_filepath, sep="\t")

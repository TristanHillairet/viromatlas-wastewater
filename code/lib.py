import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import spacy
from scipy.cluster import hierarchy
from Levenshtein import distance

def extract_infos(table,column,N=4):
    """Extraction of the values in a column of a pandas.DataFrame and plotting an histogram

    Args:
        table (pandas.DataFrame): a pandas table.
        column (String): name of the column to histogramm.
        N (int, optional): number of class to show. defaults to 4.

    Returns:
        dictionary: keys are values possible to take and values the number of occurence of the key.
    """
    # Extract the dictionary
    count  = dict(table[column].value_counts())
    null_count = table[column].isnull().sum()

    # Extract keys and values as lists
    values = list(count.values())
    keys   = list(count.keys())

    # Keep only the N first keys and create a key others
    new_keys = keys[:N]
    new_keys.append('others')
    new_keys.append('unspecified')

    # Keep only the values for the N first keys and calculate the value for the ket others
    new_values = values[:N]
    new_values.append(sum(values[N+1:]))
    new_values.append(null_count)
    new_values = new_values / sum(new_values)

    # Calculate colors on a blue scale (and black for the others key)
    bleu_min = (0,0,255)
    bleu_max = (255,255,255)
    colors = ['#%02X%02X%02X' % (random.randint(bleu_min[0], bleu_max[0]),
                                 random.randint(bleu_min[1], bleu_max[1]),
                                 random.randint(bleu_min[2], bleu_max[2])) for _ in range(len(new_keys)-2)]
    colors.append('#000000')
    colors.append('#808080')

    # Plot of the histogramm
    plt.bar(new_keys, new_values, color=colors, edgecolor='black')
    plt.xticks(rotation='vertical')
    plt.ylabel('Frequency')
    for i in range(10, 100, 10):
        plt.axhline(y=i/100, color='gray', linestyle='--', linewidth=0.5)
    plt.title(f'{column} frequency repartition')
    plt.show()

    return count

def levenshtein_dendogram(chains,ct=40):
    """Plotting and extracting dendogram of levenshtein distance of a string chain list.

    Args:
        chains (list[string]): list of strings.
        ct (int, optional): color threshold for dendogram plot. defaults to 40.

    Returns:
        dictionary: dendogram of levenshtein distance
    """
    # Levenshtein distance between strings
    dist_matrix = np.zeros((len(chains), len(chains)))
    for i in range(len(chains)):
        for j in range(len(chains)):
            dist_matrix[i, j] = distance(chains[i], chains[j])

    # Priorisation
    Z = hierarchy.linkage(dist_matrix, method='ward')

    # Plot of the dendogram
    plt.figure(figsize=(10, 5))
    dn = hierarchy.dendrogram(Z, labels=chains, color_threshold=ct)
    plt.xticks(rotation='vertical')
    plt.ylabel('Levenshtein distance')
    plt.show()

    return dn

def semantix_dendogram(chains,ct=3):
    """Plotting and extracting dendogram of semantix distance of a string chain list

    Args:
        chains (list[string]): list of strings
        ct (int, optional): color thresholds for dendogram plot. Defaults to 3.

    Returns:
        _type_: _description_
    """
    # Loading spaCy english language model
    nlp = spacy.load("en_core_web_md")

    # Semantix distance (WMD) betwenn strings
    dist_matrix = np.zeros((len(chains), len(chains)))
    for i in range(len(chains)):
        for j in range(len(chains)):
            dist_matrix[i, j] = nlp(chains[i]).similarity(nlp(chains[j]))

    # Priorisation
    Z = hierarchy.linkage(dist_matrix, method='ward')

    # Plot of teh dendogram
    plt.figure(figsize=(10, 5))
    dn = hierarchy.dendrogram(Z, labels=chains, color_threshold=3)
    plt.xticks(rotation='vertical')
    plt.ylabel('Semantix distance')
    plt.show()

    return dn

def repartition(dict,t=0.95):
    """Calculate the repartition function of the infos dictionary

    Args:
        dict (dictionary): infos dictionary of a column
        t (float, optional): threshold to show where f(X) > Y. defaults to 0.95.

    Returns:
        list[object]: f(X) where f is repartition function
    """
    # Load values from the dictionary
    values = list(dict.values())

    # Initialize tot and repartiontion list
    tot = sum(values)
    repart = [0]

    # Repartition for each X
    for i in range(len(values)-1):
        repart.append(sum(values[:i+1])/tot)

    # Id where repartition > threshold
    thr = min([i for i,val in enumerate(repart) if val > t])

    # Plot of the repartition function
    plt.plot(repart, color='blue', linewidth=3)
    plt.axvline(x=thr, color='red', linestyle='--')
    plt.xlim(0,len(repart))
    plt.ylim(0,1)
    plt.ylabel("Repartition")
    plt.show()

    return repart,thr
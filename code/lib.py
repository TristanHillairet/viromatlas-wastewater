import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import spacy
from scipy.cluster import hierarchy
from Levenshtein import distance

def hist(df_datas,width_bar=0.2,colors=[]):
    """_summary_

    Args:
        list_dict (_type_): _description_
        colors (list, optional): _description_. Defaults to [].
        dtype (str, optional): _description_. Defaults to 'quali'.
        xmin (int, optional): _description_. Defaults to 0.
        xmax (int, optional): _description_. Defaults to 100.
        ymin (int, optional): _description_. Defaults to 0.
        ymax (int, optional): _description_. Defaults to 100.
        y_gap (int, optional): _description_. Defaults to 10.
    """
    # Load keys and normalize values
    offset = -width_bar * (len(df_datas.columns)/2)
    positions = np.arange(len(df_datas.index))

    i = 0
    for col in list(df_datas.columns):
        plt.bar(positions+offset, df_datas[col], color=[colors[i]]*len(df_datas[col]), width=width_bar, label=col)
        offset += width_bar
        i += 1

    plt.xticks(positions, df_datas.index, rotation='vertical')
    plt.legend()
    return 

def norm_hist(dict,colors=[]):
    """_summary_

    Args:
        dict (_type_): _description_
        colors (list, optional): _description_. Defaults to [].
        dtype (str, optional): _description_. Defaults to 'quali'.
        xmin (int, optional): _description_. Defaults to 0.
        xmax (int, optional): _description_. Defaults to 1.
    """
    # Load keys and normalize values
    keys = list(dict.keys())
    values = list(dict.values())
    def norm(y):
        return y/sum(values)
    def unnorm(y):
        return y*sum(values)
    # Plot of the histogramm
    fig,ax1 = plt.subplots()
    ax1.bar(keys, values, color=colors, edgecolor='black')
    ax1.set_ylabel("Total")
    ax1.set_ylim(0,sum(values))
    ax1.set_xticks(keys)
    ax2 = ax1.secondary_yaxis('right', functions=(norm,unnorm))
    ax2.set_ylabel("Frequency")
    plt.show()
    return 

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
    dn = hierarchy.dendrogram(Z, labels=chains, color_threshold=ct)
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


    thr = min([i for i,val in enumerate(repart) if val > t])

    # Plot of the repartition function
    plt.plot(repart, color='blue', linewidth=3)
    plt.axvline(x=thr, color='grey', linestyle='--')
    plt.xlim(0,len(repart))
    plt.ylim(0,1)
    plt.ylabel("Repartition")
    plt.show()

    return repart,thr
### May 2024
### Tristan Hillairet
### Style and plotting graph functions script

#############################################

################
# IMPORTATIONS #
################

#
# System importations
#

import sys
import time
import itertools
import os

#
# Data management importations
#

import pandas as pd

#
# Maths importations
#

import random

#
# Graphs importations
#

import plotly.express as px

#
# Scripts | personal libraries importations
#

folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if folder_path not in sys.path:
    sys.path.insert(0, folder_path)
import globals as p

###########################
# RANDOM PALETTE CREATION #
###########################

#
# random color creation
#

def random_color(r_lim=(0,255),g_lim=(0,255),b_lim=(0,255)):
    """returns a random colour in RGB format

    Args:
        r_lim (tuple, optional): red gray level limits. Defaults to (0,255).
        g_lim (tuple, optional): green gray level limits. Defaults to (0,255).
        b_lim (tuple, optional): blue gray level limits. Defaults to (0,255).

    Returns:
        string: string colour in format 'rgb(r,g,b)'
    """
    r = random.randint(r_lim[0], r_lim[1])
    g = random.randint(g_lim[0], g_lim[1])
    b = random.randint(b_lim[0], b_lim[1])
    return f'rgb({r},{g},{b})'

#
# random palette creation
#

def random_palette(n,r_lim=(0,255),g_lim=(0,255),b_lim=(0,255)):
    """returns a random colour palette 

    Args:
        n (int): number of colours in the palette
        r_lim (tuple, optional): red gray level limits. Defaults to (0,255).
        g_lim (tuple, optional): green gray level limits. Defaults to (0,255).
        b_lim (tuple, optional): blue gray level limits. Defaults to (0,255).

    Returns:
        list[string]: list of colours in RGB format
    """
    return [random_color(r_lim=r_lim,g_lim=g_lim,b_lim=b_lim) for _ in range(n)]

############################
# STRING AND CELLS OUTPUTS #
############################

#
# errors and cells output management 
#

def print_cell_msg(string,time="?",error=True,error_type=''):
    """print a cell execution message and handle error

    Args:
        string (string): string message output
        time (float): execution time
        error (bool, optional): wether the message handle an error or not. Defaults to True.
        error_type (str, optional): handle error type. Defaults to ''.
    """
    if error:
        print(f"\033[91m#[{time}s]---{string}\n{error_type}\033[0m")
        return
    print(f"\033[92m#[{time}s]---{string}\033[0m")
    return

#
# execution message management
#

def animate_string_execution(message):
    """print an animated execution message with a thread

    Args:
        message (string): execution message
    """
    def animate_string():
        for c in itertools.cycle(['      ', '-     ', '--    ', '---   ','----  ','------ ','------']):
            if p.get_done():
                break
            sys.stdout.write(f'\r\033[94m# {message} {c}')
            sys.stdout.flush()
            time.sleep(0.5)
    return animate_string

#########################
# GRAPHS PLOT FUNCTIONS #
#########################

#
# TreeMap plot
#

def TreeMap(df,col,parent_col='',width=800,height=600,r_lim=(0,255),g_lim=(0,255),b_lim=(0,255),fix_seed=0):
    """plot a TreeMap from a pandas.DataFrame with plotly

    Args:
        df (_type_): a dataframe
        col (_type_): a column from the dataframe
        parent_col (optional): possible parents in the TreeMap. Defaults to ''.
        width (int, optional): width of the graph plot. Defaults to 800.
        height (int, optional): height of the graph plot. Defaults to 600.
        r_lim (tuple, optional): red gray level limits. Defaults to (0,255).
        g_lim (tuple, optional): green gray level limits. Defaults to (0,255).
        b_lim (tuple, optional): blue gray level limits. Defaults to (0,255).
    """

    # get unique values in col
    unique_val = list(df[col].unique())

    if parent_col != '':

        # initialize parents and sizes of each value in col
        parents = [df.loc[df[col] == val, parent_col].values[0] for val in unique_val]
        sizes = [len(df[df[col] == val]) for val in unique_val]

        # initialize a dataframe for the TreeMap
        treemap_pd = pd.DataFrame(
            dict(col=unique_val,parents=parents,sizes=sizes)
        )

        # create "random" color palette
        color_map = {'(?)':'lightgrey'}
        if fix_seed != 0:
            random.seed(fix_seed)
        colors = random_palette(len(parents),r_lim=r_lim,g_lim=g_lim,b_lim=b_lim) # random color palette
        color_map.update(dict(zip(treemap_pd['parents'].unique(),colors)))

        # plot the TreeMap
        treemap_pd["all"] = "all"
        fig = px.treemap(treemap_pd,
                path=['all','parents','col'], 
                values='sizes', 
                color='parents', # color parents
                color_discrete_map=color_map
        )
        fig.update_traces(root_color="lightgrey")
        fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
        fig.update_layout(
            width=width, # width of the graph
            height=height# height of the graph
        )
        fig.show()
    
    else:

        # initialize sizes of each values in col
        sizes = [len(df[df[col] == val]) for val in unique_val]

        # initialize a dataframe for the TreeMap
        treemap_pd = pd.DataFrame(
            dict(col=unique_val,sizes=sizes)
        )

        # plot the TreeMap
        treemap_pd["all"] = "all"
        fig = px.treemap(treemap_pd,
                path=['all','col'], 
                values='sizes',
        )
        fig.update_traces(root_color="lightgrey")
        fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
        fig.update_layout(
            width=width, # width of the graph
            height=height# height of the graph
        )
        fig.show()
    
    return

#
# Repartition function plot
#

def RepartitionFunction(df,col,y_type='norm',y_thr=0,width=800,heigth=600):
    """plot a repartition function from a pandas.DataFrame column with plotly

    Args:
        df (_type_): a dataframe
        col (_type_): a column from the dataframe
        y_type (str, optional): type of y axis. Defaults to 'norm'. Either 'norm' or 'tot'
        y_thr (int, optional): y threshold to split graph. Defaults to 0. if set returns thr where f(thr) = y_thr
        width (int, optional): width of the graph plot. Defaults to 800.
        heigth (int, optional): height of the graph plot. Defaults to 600.
    """

    # get unique values and their count in col
    value_count = dict(df[col].value_counts())
    keys   = [i for i in range(len(value_count))]
    values = list(value_count.values())

    # initialize tot and repartition list
    if y_type == 'norm':
        tot = sum(values)
    elif y_type == 'tot':
        tot = 1
    else:
        print("PLOT ERROR FOR RepartitionFunction : y_type must be 'norm' or 'tot'")
    repart = [0]

    # repartition for each X
    for i in range(len(values)-1):
        repart.append(sum(values[:i+1])/tot)

    # initialize dataframe for Plot
    plot_df = pd.DataFrame(
        dict(keys=keys,values=repart)
    )

    # plot the repartition function
    fig = px.line(plot_df, x="keys", y="values", title='Fonction de répartition')

    # plot the x value where F(x) > y_thr if y_thr != 0
    if y_thr != 0:
        thr = min([i for i,val in enumerate(repart) if val > y_thr])
        fig.add_vline(x=thr, line_width=3, line_dash="dash", line_color="lightcoral")
        fig.update_layout(
        width=width, # width of the graph
        height=heigth# heigth of the graph
        )
        fig.show()
        return thr

    # update layout plot
    fig.update_layout(
        width=width, # width of the graph
        height=heigth# heigth of the graph
    )
    fig.show()

    return

#
# Histogram plot
# 

def Histogram(df,col,color_col='',width=800,height=600,r_lim=(0,255),g_lim=(0,255),b_lim=(0,255),fix_seed=0):
    """plot the histogram of a pandas.DataFrame column with plotly

    Args:
        df (_type_): a dataframe
        col (_type_): a column from the dataframe
        color_col (optional): a column from the dataframe to color bars of the histogram. Defaults to ''.
        width (int, optional): width of the graph plot. Defaults to 800.
        height (int, optional): height of the graph plot. Defaults to 600.
        r_lim (tuple, optional): red gray level limits. Defaults to (0,255).
        g_lim (tuple, optional): green gray level limits. Defaults to (0,255).
        b_lim (tuple, optional): blue gray level limits. Defaults to (0,255).
    """

    if color_col != '':

        # create "random" color palette
        color_map = {'(?)':'lightgrey'}
        if fix_seed != 0:
            random.seed(fix_seed)
        colors = random_palette(len(list(df[color_col].unique())),r_lim=r_lim,g_lim=g_lim,b_lim=b_lim) # random color palette
        color_map.update(dict(zip(df[color_col].unique(),colors)))

        # plot if there is a color column
        fig = px.histogram(
            df, 
            x=col, 
            color=color_col,
            color_discrete_map=color_map
        )

    else:

        # plot an histogram without a specified color column
        fig = px.histogram(
            df, 
            x=col,
        )
    
    # update graph layout
    fig.update_layout(
        width=width, # width of the graph
        height=height# height of the graph
    )            
    fig.show()

    return

####################
# SCRIPT EXECUTION #
####################

if __name__ == "__main__":

    help = str(input(f"\033[92m#---Graph function and styles library (if help enter h)\033[0m : "))

    if help == "h":

        print("\033[92m#---random_color()\033[0m")
        print("\033[92m#---random_palette()\033[0m")
        print("\033[92m#---print_cell_msg()\033[0m")
        print("\033[92m#---animate_string_execution()\033[0m")
        print("\033[92m#---TreeMap()\033[0m")
        print("\033[92m#---RepartitionFunction()\033[0m")
        print("\033[92m#---Histogram()\033[0m")
        

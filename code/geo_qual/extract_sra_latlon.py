### May 2024
### Tristan Hillairet
### Parsing GeoSRA metadatas to get georef metadatas    

####################################################

################
# IMPORTATIONS #
################

#
# System importations
#

import time
import datetime
import threading
import re
import multiprocessing
import sys
import os

#
# Data management importations
#

import pandas as pd
import dask.dataframe as ddf
import json
from pyarrow import fs

#
# Maths importations
# 

import numpy as np
from dateutil.parser import parse as dateparse
from lat_lon_parser import parse as parse
import reverse_geocode as rg
import pycountry

#
# Scripts | personal libraries importations
#

folder1_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.', 'representations'))
if folder1_path not in sys.path:
    sys.path.insert(0, folder1_path)
import graph_lib as g

folder2_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if folder2_path not in sys.path:
    sys.path.insert(0, folder2_path)
import globals as p

##############
# PARAMETERS #
##############

#
# distant path to file
#

distant_path = 'geo-sra/samples/sra_metadata_toy.parquet'

#
# local path to file
#

local_path   = "/Users/tpietav/Desktop/data/raw/sra_metadata_toy.parquet"

#
# local output path file
# 

local_output_path = "/Users/tpietav/Desktop/data/processed/sra_metadata_toy_geoqual.parquet"

####################
# HELPER FUNCTIONS #
####################

#
# check if string is date
#

def is_date(string, fuzzy=False):
    """returns if a string is a date or not

    Args:
        string (string): string to check
        fuzzy (bool, optional): fuzzy string or not. Defaults to False.

    Returns:
        boolean: if the string is a date or not
    """
    try: 
        dateparse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False

#
# select collection date
#

def select_first_collection_date(x):
    """select the first collection date of object x

    Args:
        x (pd.Series): series of collection date issue from a dataframe

    Returns:
        string: first collection date given for x
    """
    if not pd.isnull(x):
        return x[0]
    else:
        return None

#
# check if a string is a full alpha string
#

def is_alpha_string(string,max_words=2):
    """return wether a string is an alpha string or not

    Args:
        string (string): string to check 
        max_words (int, optional): max words that can contain the string. Defaults to 2.
    """
    for n in range(max_words):
        pattern = r'^[a-zA-Z]+ {'+str(n)+r'}[a-zA-Z]+$'
        if re.match(pattern,string):
            return True
    
    return False

#
# give decimal precision of a number
#

def decimal_precision(nombre):
    """return the decimal precision of a number

    Args:
        nombre (float): number to calc

    Returns:
        int: number of decimals given in the number
    """
    if nombre == np.NaN:
        return np.NaN
    chaine = str(nombre)
    index_point = chaine.find('.')
    if index_point == -1:
        return np.NaN
    return len(chaine) - index_point - 1

#
# check if a string country is Na
#

def country_is_na(country):
    """return wether a country field is NA or not

    Args:
        country (string): Country value

    Returns:
        bool: True if the string is Na False if not
    """
    country_na_tag = ['uncalculated','<NA>']
    if pd.isna(country):
        return True
    elif country in country_na_tag:
        return True
    return False

#
# check if 2 strings of countries matches
#

def match_countries(country_name_1,country_name_2):
    """say if two string of countries matches or not

    Args:
        country_name_1 (string): string name of country one
        country_name_2 (string): string name of country two

    Returns:
        bool: True if the two countries are the same
    """
    alpha_country_1 = country_name_to_alpha_3.get(country_name_1, None)
    alpha_country_2 = country_name_to_alpha_3.get(country_name_2, None)
    if alpha_country_1 == alpha_country_2:
        return True
    return False

#
# check if reversig latlon is better
#

def reverse_lat_lon_test(longitude,latitude,country):
    """say if reversing coodinates matches the country given

    Args:
        longitude (float): longitude of the point
        latitude (float): latitude of the point
        country (string): string name of the country given

    Returns:
        bool: True if reversing coordinates matches the country
    """
    coordinates = [(longitude,latitude)]
    alpha3_country = country_name_to_alpha_3.get(country, None)
    rgeocode = rg.search(coordinates)          
    if(len(rgeocode)>0):
        rg_country = rgeocode[0]['country']
        alpha3_rg_country = country_name_to_alpha_3.get(rg_country, None)
    if alpha3_country == alpha3_rg_country:
        return True
    return False

#
# parse geoSra datas and calculate GEO_QUAL
#

def parse_geo_sra(pd_array_partition): 

    array_result = []
    #i,n = 0,len(pd_array_partition)

    for samples in pd_array_partition.itertuples(index=False):

        # initialise variable

        lat_lon_src = ''
        lat_lon_raw = ''
        lat_src = ''
        lat_raw = ''

        lon_src = ''
        lon_raw = ''
        
        lat_lon = ''
        lat = ''
        lon = ''
        lat_list = None
        lon_list = None
        lat_lon_list = None

        latitude = None
        longitude = None

        latitude_precision = None
        longitude_precision = None

        has_latlon = False

        geo_qual = 4.0

        #
        # parse from metadata available
        #

        bioproject = samples.bioproject
        acc = samples.acc
        organism = samples.organism
        assay_type = samples.assay_type
        instrument = samples.instrument
        librarylayout = samples.librarylayout
        libraryselection = samples.libraryselection
        librarysource = samples.librarysource
        geo_loc_name_country_calc = samples.geo_loc_name_country_calc
        geo_loc_name_country_continent_calc = samples.geo_loc_name_country_continent_calc
        mbytes = samples.mbytes
        mbases = samples.mbases

        releasedate = None
        if( not pd.isnull(samples.releasedate)) :
            if(is_date(samples.releasedate)):
                 releasedate = pd.to_datetime(samples.releasedate)

        collection_date = None
        if( not pd.isnull(samples.collection_date)) :
            # multiple collection date are sometimes available for the same sample
            # take the first one
            if(is_date(samples.collection_date)):
                collection_date = pd.to_datetime(samples.collection_date)

        # 
        # reverse geocoding
        #

        rg_country_code = ''
        rg_city = ''
        rg_country = ''

        #
        # parse from attributes metadata available
        #

        lat_lon_patterns = [
            r"(-?\d+\.\d+ [NS]) (-?\d+\.\d+ [WE])",   # xx.xxx N xx.xxx W
            r"(-?\d+\.\d+' [NS]) (-?\d+\.\d+' [WE])", # xx.xxx' N xx.xxx' W
            r"(-?\d+\.\d+° [NS]) (-?\d+\.\d+° [WE])", # xx.xxx° N xx.xxx° W
            r'(-?\d+ [NS]) (-?\d+ [WE])',             # xx N xx W
            r'(-?\d+\.\d+ [NS]) (-?\d+ [WE])',        # xx.xxx N xx W
            r'(-?\d+ [NS]) (-?\d+\.\d+ [WE])',        # xx N xx.xxx W 
            r'(-?\d+\.\d+) (-?\d+\.\d+)',             # xx.xxx xx.xxx
            r'(-?\d+\.\d+°) (-?\d+\.\d+°)',           # xx.xxx° xx.xxx°
            r"(-?\d+\.\d+') (-?\d+\.\d+')",           # xx.xxx' xx.xxx'
        ]

        lat_patterns = [
            r"(-?\d+\.\d+ [NS])",   # xx.xxx N xx.xxx W
            r"(-?\d+\.\d+' [NS])",  # xx.xxx' N xx.xxx' W
            r"(-?\d+\.\d+° [NS])",  # xx.xxx° N xx.xxx° W
            r'(-?\d+ [NS])',             # xx N xx W
            r'(-?\d+\.\d+ [NS])',        # xx.xxx N xx W
            r'(-?\d+\.\d+)',             # xx.xxx xx.xxx
            r'(-?\d+\.\d+°)',           # xx.xxx° xx.xxx°
            r"(-?\d+\.\d+')",           # xx.xxx' xx.xxx'
        ]

        lon_patterns = [
            r"(-?\d+\.\d+ [WE])",
            r"(-?\d+\.\d+' [WE])",
            r"(-?\d+\.\d+° [WE])",
            r"(-?\d+ [WE])",
            r"(-?\d+\.\d+ [WE])",
            r"(-?\d+\.\d+)",
            r"(-?\d+\.\d+°)",
            r"(-?\d+\.\d+')"
        ]

        ###########################################################################
        # 
        # selected tag
        #

        lat_lon_tag = ['lat_lon_sam_s_dpl34','lat_lon_sam_s_dpl1','geographic_location__latitude_and_longitude__sam','geographic_location__latitudeandlongitude__sam','latitude__and_longitude_sam','latitude_and_longitude_sam','lattitude_and_logitude_sam','lat_lon_sam','latlon_sam','location_coordinates_sam','other_gps_coordinates_sam','lat_lon_dms_sam','lat_long_correct_sam','lat_lon_run']

        lat_tag = ['geographic_location__latitude__sam_s_dpl4','lat_lon_sam_s_dpl1','latitude_sam','lat_sam','geographic_location__latitude__sam','geographiclocation_latitude__sam','latitude_dd_sam','latitude_deg_sam','biological_material_latitude_sam']

        lon_tag = ['geographic_location__longitude__sam_s_dpl5','longitude_sam','lon_sam','geographic_location__longitude__sam','longitude_dd_sam','longitude_deg_sam','biological_material_longitude_sam']
        
        filtered_tag = np.concatenate((lat_lon_tag,lat_tag,lon_tag))

        # Problem of serialization between dask partition / pandas
        # attributes = samples.attributes
        # attributes_df = pd.DataFrame.from_records(attributes)


        attributes = samples.jattr

        if(not attributes==""):

            attributes_df = pd.DataFrame(json.loads(attributes).items(),columns=['k', 'v'])

        else:
            attributes_df=pd.DataFrame()
            print("ERROR EMPTY")

        ###########################################################################
        
        if('k' in attributes_df):
            
            attributes_df = attributes_df.loc[attributes_df['k'].isin(filtered_tag)]

            for index, attribute in attributes_df.iterrows():
                
                #store attributes keyword

                #if attribute['k'] in dict_attributes_k :
                #    dict_attributes_k[attribute['k']] = dict_attributes_k[attribute['k']] + 1
                #else :
                #    dict_attributes_k[attribute['k']] = 0
                #    print("keyword", attribute['k'])

                ###########################################################################
                # latitude - longitude in the same field


                if attribute['k'] in lat_lon_tag :
                    
                    lat_lon_src = attribute['k']

                    if(isinstance(attribute['v'], list)):
                        lat_lon = attribute['v'][0]
                        lat_lon_raw = attribute['v'][0]
                    else:
                        lat_lon = attribute['v']
                        lat_lon_raw = attribute['v']

                    if not is_alpha_string(str(lat_lon)):
                        for pattern in lat_lon_patterns:
                            if re.match(pattern,str(lat_lon)):
                                lat_lon_list = re.match(pattern,str(lat_lon))

                ###########################################################################

                ###########################################################################
                # latitude - longitude in separated fields

                # latitude

                if attribute['k'] in lat_tag :
                    
                    lat_src = attribute['k']

                    if(isinstance(attribute['v'], list)):
                        lat = attribute['v'][0]
                        lat_raw = attribute['v'][0]
                    else:
                        lat = attribute['v']
                        lat_raw = attribute['v']

                    if lat_lon_src == '':
                        lat_lon_src = lat_src
                        lat_lon_raw = lat_raw
                    else:
                        lat_lon_src = lat_src + " " + lat_lon_src
                        lat_lon_raw = str(lat_raw) + " " + lat_lon_raw

                    if not is_alpha_string(str(lat)) :
                        for pattern in lat_patterns:
                            if re.match(pattern,str(lat)):
                                lat_list = re.match(pattern,str(lat))

                # longitude
                
                if attribute['k'] in lon_tag :
                    
                    lon_src = attribute['k']
                    
                    if(isinstance(attribute['v'], list)):
                        lon = attribute['v'][0]
                        lon_raw = attribute['v'][0]
                    else:
                        lon = attribute['v']
                        lon_raw = attribute['v']
                    
                    if lat_lon_src == '':
                        lat_lon_src = lon_src
                        lat_lon_raw = lon_raw
                    else:
                        lat_lon_src = lat_lon_src + " " + lon_src
                        lat_lon_raw = lat_lon_raw + " " + str(lon_raw)

                    if not is_alpha_string(str(lon)) :
                        for pattern in lon_patterns:
                            if re.match(pattern,str(lon)):
                                lon_list = re.match(pattern,str(lon))
                
                ###########################################################################

            ###########################################################################
                
            #
            # lat_lon_parser
            #

            if lat_lon_list is not None :

                try :
                    latitude = parse(lat_lon_list.group(1))
                    longitude = parse(lat_lon_list.group(2))
                    latitude_precision = decimal_precision(latitude)
                    longitude_precision = decimal_precision(longitude)
                except:
                    print("parse ERROR", acc,lat_lon)
                    pass

            if (lon_list is not None) & (lat_list is not None) & (str(lon).count(".") <=1) & (str(lat).count(".") <=1) :

                if (lon_list.group(1) is not None) & (lat_list.group(1) is not None) :

                    try :
                        latitude = parse(lat_list.group(1))
                        longitude = parse(lon_list.group(1))
                        latitude_precision = decimal_precision(latitude)
                        longitude_precision = decimal_precision(longitude)
                    except :
                        print("parse ERROR", acc,lat, lon)
                        pass

            #
            # reverse geocoding
            #
            
            if((latitude is not None) & (longitude is not None)):
                coordinates = [(latitude,longitude)]
                has_latlon=True
                rgeocode = rg.search(coordinates)
                
                if(len(rgeocode)>0):
                    rg_country_code = rgeocode[0]['country_code']
                    rg_city = rgeocode[0]['city']
                    rg_country = rgeocode[0]['country']
            
            #
            # GEO_QUAL calc
            #
            
            thr_prec = 2
            if ((latitude is not None) & (longitude is not None)):
                if ((latitude_precision > thr_prec) | (longitude_precision > thr_prec)):
                    geo_qual = 1.3
                    if not country_is_na(geo_loc_name_country_calc):
                        geo_qual = 1.2
                        if match_countries(geo_loc_name_country_calc,rg_country):
                            geo_qual = 1.1
                        else:
                            if reverse_lat_lon_test(latitude,longitude,geo_loc_name_country_calc):
                                longitude,latitude = latitude,longitude
                                geo_qual = 1.15
                else:
                    geo_qual = 2.3
                    if not country_is_na(geo_loc_name_country_calc):
                        geo_qual = 2.2
                        if match_countries(geo_loc_name_country_calc,rg_country):
                            geo_qual = 2.1
                        else:
                            if reverse_lat_lon_test(latitude,longitude,geo_loc_name_country_calc):
                                longitude,latitude = latitude,longitude
                                geo_qual = 2.15

            elif (((latitude is None) | (longitude is None)) & (not country_is_na(geo_loc_name_country_calc))):
                geo_qual = 3.0
            
        # append result
        array_result.append([bioproject,acc,organism,assay_type,instrument,librarylayout,libraryselection,librarysource,geo_loc_name_country_calc,geo_loc_name_country_continent_calc,mbytes,mbases,releasedate,collection_date,lat_lon_src,lat_lon_raw,latitude,longitude,latitude_precision,has_latlon,longitude_precision,rg_country_code,rg_city,rg_country,geo_qual])

    columns = ['bioproject', 'acc','organism','assay_type','instrument','librarylayout','libraryselection','librarysource','geo_loc_name_country_calc','geo_loc_name_country_continent_calc','mbytes','mbases','releasedate', 'collection_date','lat_lon_src','lat_lon_raw','latitude','longitude','latitude_precision','has_latlon','longitude_precision','rg_country_code','rg_city','rg_country','GEO_QUAL']
    dtype = [str,str,str,str,str,str,str,str,str,str,int,int,datetime.date,datetime.date,str,str,float,float,float,float,bool,str,str,str,float,int,int,str]
    cdt={i[0]: i[1] for i in zip(columns, dtype)}        
    pd_result = pd.DataFrame(data=array_result,columns=list(cdt))

    return pd_result

####################
# SCRIPT EXECUTION #
####################

if __name__ == "__main__":

    help = str(input(f"\033[92m#---Parsing GeoSRA metadatas to get georef metadatas (if help enter h)\033[0m : "))

    if help == "h":

        print("\033[92m#---is_date()\033[0m")
        print("\033[92m#---select_first_collection_date()\033[0m")
        print("\033[92m#---is_alpha_string()\033[0m")
        print("\033[92m#---decimal_precision()\033[0m")
        print("\033[92m#---country_is_na()\033[0m")
        print("\033[92m#---match_countries()\033[0m")
        print("\033[92m#---reverse_lat_lon_test()\033[0m")
        print("\033[92m#---parse_geo_sra()\033[0m")
        sys.exit()

    #####################
    # METADATAS OPENING #
    #####################

    p.set_done(False)
    t1 = time.time()

    f = g.animate_string_execution("Exécution du code")
    thr = threading.Thread(target=f)
    thr.start()

    try:
    
        s3fs = fs.S3FileSystem(
            endpoint_override="localhost:9000",
            access_key="lbIwWP4FIBtjBwU5AaI0",
            secret_key="nw54mjkG3CjxjvkyqxACTbYOuhtzyc1YmkMaJXeL",
            scheme="http"
        )
        df = pd.read_parquet(distant_path,filesystem=s3fs)

        p.set_done(True)

        type_open = "dist"

        print('')
        t2 = time.time()
        t  = np.round(t2 - t1,decimals=1)
        g.print_cell_msg("Ouverture à distance réussie",time=t,error=False)

    except Exception as e:

        type_open = "loc"
    
        print('')
        t2 = time.time()
        t  = np.round(t2 - t1,decimals=1)
        g.print_cell_msg("Erreur d'ouverture à distance",time=t,error_type=e)

        try:

            df = pd.read_parquet(local_path)

            p.set_done(True)

            print('')
            t2 = time.time()
            t  = np.round(t2 - t1,decimals=1)
            g.print_cell_msg("Ouverture locale réussie",time=t,error=False)
    
        except Exception as e_local:

            p.set_done(True)

            print('')
            t2 = time.time()
            t  = np.round(t2 - t1,decimals=1)
            g.print_cell_msg("Erreur d'ouverture locale",time=t,error_type=e_local)

    ############################################
    # DICTIONNARY OF COUNTRIES:ALPHA3 CREATION #
    ############################################

    t1 = time.time()

    try:

        country_name_to_alpha_3 = {country.name: country.alpha_3 for country in pycountry.countries}

        additional_mappings = {
            "Svalbard and Jan Mayen":"NOR",
            "South Korea":"KOR",
            "Czech Republic":"CZE",
            "Russian Federation":"RUS",
            "Russia":"RUS",
            "Hong Kong":"CHN",
            "Vietnam":"VNM",
            "Taiwan":"TWN",
            "Palestinian Territory":"ISR",
            "Turkey":"TUR",
            "Bolivia":"BOL",
            "Libyan Arab Jamahiriya":"LBY",
            "Cape Verde":"CPV",
            "Aland Islands":"ALA",
            "Saint Helena":"SHN",
            "Democratic Republic of the Congo":"COD",
            "Laos":"LAO",
            "Tanzania":"TZN",
            "Brunei":"BRN",
            "Iran":"IRN",
            "Svalbard":"NOR",
            "Macau":"CHN",
        }       
        country_name_to_alpha_3.update(additional_mappings)

        t2 = time.time()
        t  = np.round(t2 - t1,decimals=1)
        g.print_cell_msg("Création du dictionnaire des pays réussie",time=t,error=False)

    except Exception as e:
    
        t2 = time.time()
        t  = np.round(t2 - t1,decimals=1)
        g.print_cell_msg("Erreur de création du dictionnaire des pays",time=t,error_type=e)
    
    ###################
    # COLLECTION DATE #
    ###################

    t1 = time.time()

    try:

        df["collection_date"] = df['collection_date_sam'].apply(select_first_collection_date)
    
        t2 = time.time()
        t  = np.round(t2 - t1,decimals=1)
        g.print_cell_msg("Calcul de collection_date",time=t,error=False)

    except Exception as e:

        t2 = time.time()
        t  = np.round(t2 - t1,decimals=1)
        g.print_cell_msg("Erreur de calcul de collection_date",time=t,error_type=e)
    
    ###########################
    # DASK DATAFRAME CREATION #
    ###########################

    t1 = time.time()
    
    p.set_done(False)
    thr = threading.Thread(target=f)
    thr.start()

    try:

        num_partitions = multiprocessing.cpu_count()
        dask = ddf.from_pandas(df, npartitions=num_partitions-5)

        p.set_done(True)

        print('')
        t2 = time.time()
        t  = np.round(t2 - t1,decimals=1)
        g.print_cell_msg("Création du dask dataframe",time=t,error=False)

    except Exception as e:

        p.set_done(True)

        print('')
        t2 = time.time()
        t  = np.round(t2 - t1,decimals=1)
        g.print_cell_msg("Erreur de création du dask dataframe",time=t,error_type=e)
    
    #####################################
    # FINAL METADATAS DATAFRAME COLUMNS #
    #####################################

    t1 = time.time()

    try:

        columns = ['bioproject', 'acc','organism','assay_type','instrument','librarylayout','libraryselection','librarysource','geo_loc_name_country_calc','geo_loc_name_country_continent_calc','mbytes','mbases','releasedate', 'collection_date','lat_lon_src','lat_lon_raw','latitude','longitude','latitude_precision','has_latlon','longitude_precision','rg_country_code','rg_city','rg_country','GEO_QUAL']
        dtype = [str,str,str,str,str,str,str,str,str,str,int,int,datetime.date,datetime.date,str,str,float,float,float,float,bool,str,str,str,float,int,int,str]
        cdt={i[0]: i[1] for i in zip(columns, dtype)}
        meta_df = pd.DataFrame(columns=list(cdt))

        t2 = time.time()
        t  = np.round(t2 - t1,decimals=1)
        g.print_cell_msg("Création du dataframe des métadonnées",time=t,error=False)

    except Exception as e:

        t2 = time.time()
        t  = np.round(t2 - t1,decimals=1)
        g.print_cell_msg("Erreur de création du dataframe des métadonnées",time=t,error_type=e)
    
    #################
    # PARSING DATAS #
    #################

    t1 = time.time()

    p.set_done(False)
    thr = threading.Thread(target=f)
    thr.start()

    try:

        pd_output = dask.map_partitions(parse_geo_sra, meta=meta_df).compute(scheduler='multiprocessing')

        p.set_done(True)

        print('')
        t2 = time.time()
        t  = np.round(t2 - t1,decimals=1)
        g.print_cell_msg("Exécution de parse_geo_sra réussie",time=t,error=False)
    
    except Exception as e:

        p.set_done(True)
    
        print('')
        t2 = time.time()
        t  = np.round(t2 - t1,decimals=1)
        g.print_cell_msg("Erreur d'exécution de parse_geo_sra",time=t,error_type=e)
    
    ####################
    # SAVING METADATAS #
    ####################

    t1 = time.time()

    p.set_done(False)
    thr = threading.Thread(target=f)
    thr.start()

    try:

        if type_open == "dist":
            print('to do')
    
        elif type_open == "loc":
            pd_output.to_parquet(local_output_path, index=False)
    
        p.set_done(True)
    
        print('')
        t2 = time.time()
        t  = np.round(t2 - t1,decimals=1)
        g.print_cell_msg("Enregistrement réussi",time=t,error=False)

    except Exception as e:
    
        p.set_done(True)

        print('')
        t2 = time.time()
        t  = np.round(t2 - t1,decimals=1)
        g.print_cell_msg("Erreur d'enregistrement",time=t,error_type=e)
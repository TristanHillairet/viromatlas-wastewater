import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import decimal
import re
import numpy as np

from lat_lon_parser import parse as parse

from datetime import datetime
from dateutil.parser import parse as dateparse

import reverse_geocode as rg

import json

#
# Helper FUNCTION
#

def decimal_precision(nombre):
    """
    Return decimal precision of a decimal number

    Args:
        nombre : A number 

    Returns:
        int: Number of decimals 
    """
    if nombre == np.NaN:
        return np.NaN
    chaine = str(nombre)
    index_point = chaine.find('.')
    if index_point == -1:
        return np.NaN
    return len(chaine) - index_point - 1


def column(matrix, i):
    return [row[i] for row in matrix]

def select_first_collection_date(x):
    if not pd.isnull(x):
        return x[0]
    else:
        return None
    
def is_alpha_string(string,max_words=2):
    """
    Return wether a string is an alpha string or not

    Args:
        string (String): string to check 
        max_words (int, optional): max words that can contain the string. Defaults to 2.
    """
    for n in range(max_words):
        pattern = r'^[a-zA-Z]+ {'+str(n)+r'}[a-zA-Z]+$'
        if re.match(pattern,string):
            return True
    
    return False

def country_is_na(country):
    """
    Return wether a country field is NA or not

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

def is_date(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try: 
        dateparse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False
#
# parse_geo_sra FUNCTION
#

def parse_geo_sra(pd_array_partition): 

    array_result = []

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

        geo_qual = 4

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

        #print("DATE",acc,collection_date,releasedate)

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
                    #print("parse ERROR", acc,lat_lon)
                    pass

            if (lon_list is not None) & (lat_list is not None) & (str(lon).count(".") <=1) & (str(lat).count(".") <=1) :

                if (lon_list.group(1) is not None) & (lat_list.group(1) is not None) :

                    try :
                        latitude = parse(lat_list.group(1))
                        longitude = parse(lon_list.group(1))
                        latitude_precision = decimal_precision(latitude)
                        longitude_precision = decimal_precision(longitude)
                    except :
                        #print("parse ERROR", acc,lat, lon)
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
                    geo_qual = 1
                else:
                    geo_qual = 2
            elif (((latitude is None) | (longitude is None)) & (not country_is_na(geo_loc_name_country_calc))):
                geo_qual = 3
            
        # append result
        array_result.append([bioproject,acc,organism,assay_type,instrument,librarylayout,libraryselection,librarysource,geo_loc_name_country_calc,geo_loc_name_country_continent_calc,mbytes,mbases,releasedate,collection_date,lat_lon_src,lat_lon_raw,latitude,longitude,latitude_precision,has_latlon,longitude_precision,rg_country_code,rg_city,rg_country,geo_qual])

    columns = ['bioproject', 'acc','organism','assay_type','instrument','librarylayout','libraryselection','librarysource','geo_loc_name_country_calc','geo_loc_name_country_continent_calc','mbytes','mbases','releasedate', 'collection_date','lat_lon_src','lat_lon_raw','latitude','longitude','latitude_precision','has_latlon','longitude_precision','rg_country_code','rg_city','rg_country','GEO_QUAL']
    dtype = [str,str,str,str,str,str,str,str,str,str,int,int,datetime.date,datetime.date,str,str,float,float,float,float,bool,str,str,str,int]
    cdt={i[0]: i[1] for i in zip(columns, dtype)}        
    pd_result = pd.DataFrame(data=array_result,columns=list(cdt))

    return pd_result


if __name__ == '__main__':

    import time

    st = time.time()

    #############################################################################
    #
    # Configure MinIO S3FileSystem
    #
    #############################################################################

    from minio import Minio
    from pyarrow import fs

    client = Minio(
            "134.214.213.93:9000",
            secure=False,
            access_key="lbIwWP4FIBtjBwU5AaI0",
            secret_key="nw54mjkG3CjxjvkyqxACTbYOuhtzyc1YmkMaJXeL",
        )

    client.bucket_exists("geo-sra")

    s3 = fs.S3FileSystem(endpoint_override="134.214.213.93:9000",access_key="lbIwWP4FIBtjBwU5AaI0",secret_key="nw54mjkG3CjxjvkyqxACTbYOuhtzyc1YmkMaJXeL",scheme="http")
    
    #
    #############################################################################


    #############################################################################
    #
    # Load SRA metadata .parquet from s3 bucket
    #
    #############################################################################

    print('#START - Load data')

    selected_columns = ['bioproject','acc','organism','assay_type','instrument','librarylayout','libraryselection','librarysource','geo_loc_name_country_calc','geo_loc_name_country_continent_calc','mbytes','mbases','collection_date_sam','releasedate','jattr']    

    pq_array = pq.read_table("geo-sra/sra/metadata/",columns=selected_columns, filesystem=s3)
    #pd_array = pq_array.slice(length=1000).to_pandas(split_blocks=True, self_destruct=True)
    
    pd_array = pq_array.to_pandas(split_blocks=True, self_destruct=True)
    del pq_array

    pd_array['collection_date']=pd_array['collection_date_sam'].apply(select_first_collection_date)
    

    #pd_array = pd.read_parquet("geo-sra/sra/metadata/",columns=selected_columns, filesystem=s3, engine="pyarrow")

    print('#END - Load data')

    #
    #############################################################################

    #############################################################################
    #
    # Partition and Loop over samples using dask and multiprocess
    #
    #############################################################################

    import multiprocessing
    import dask.dataframe as ddf

    num_partitions = multiprocessing.cpu_count()

    print('#START - Parse data')
    print('num_partitions',num_partitions,'----')

    dask_array = ddf.from_pandas(pd_array, npartitions=num_partitions-5)

    # dask multiprocessing

    columns = ['bioproject', 'acc','organism','assay_type','instrument','librarylayout','libraryselection','librarysource','geo_loc_name_country_calc','geo_loc_name_country_continent_calc','mbytes','mbases','releasedate', 'collection_date','lat_lon_src','lat_lon_raw','latitude','longitude','latitude_precision','has_latlon','longitude_precision','rg_country_code','rg_city','rg_country']
    dtype = [str,str,str,str,str,str,str,str,str,str,int,int,datetime.date,datetime.date,str,str,float,float,float,float,bool,str,str,str]
    cdt={i[0]: i[1] for i in zip(columns, dtype)}
    meta_df = pd.DataFrame(columns=list(cdt))

    pd_output = dask_array.map_partitions(parse_geo_sra, meta=meta_df).compute(scheduler='multiprocessing')
    #pd_output = parse_geo_sra(pd_array)

    print('#END - Parse data')

    #
    #############################################################################
    

    #pd_dict_k = pd.DataFrame.from_dict(data = dict_attributes_k, orient='index')


    #############################################################################
    # 
    # Remove row duplicates due to multiple data storage location columns (AWS, Google ...)
    #
    #############################################################################
    
    print('#START - Remove duplicates')

    pd_output.drop_duplicates(subset=['acc'],inplace=True)
    
    print('#END - Remove duplicates')

    #
    #############################################################################
    
    #############################################################################
    #
    # Add country annotation from geonamescache
    #
    #############################################################################
    
    import geonamescache

    # add continentcode information
    gc = geonamescache.GeonamesCache(1000)
    countries = gc.get_countries()
    countries_pd = pd.DataFrame.from_records(countries).T
    pd_output = pd_output.merge(countries_pd[['name','continentcode']],left_on='rg_country' , right_on='name',how='left')
    
    # add continent information
    continents = gc.get_continents()
    continents_pd = pd.DataFrame.from_records(continents).T
    pd_output = pd_output.merge(continents_pd[['continentCode','asciiName']],left_on='continentcode' , right_on='continentCode',how='left')
    pd_output.rename(columns={'asciiName':'gc_continent','population':'gc_city_population','continentcode':'gc_continent_code'}, inplace=True)

    # add city information
    # pb with city homonym
    # TO DO solve with spatial merging
    
    # cities = gc.get_cities()
    # cities_pd = pd.DataFrame.from_records(cities).T
    # pd_output = pd_output.merge(cities_pd[['name','population']],left_on='rg_city' , right_on='name',how='left')

    #############################################################################
    print('#START - Write geo-sra.parquet to s3 minio bucket')

    SCHEMA = pa.schema([
    pa.field('bioproject', pa.string()),
    pa.field('acc', pa.string()),
    pa.field('organism', pa.string()),
    pa.field('assay_type', pa.string()),
    pa.field('instrument', pa.string()),
    pa.field('librarylayout', pa.string()),
    pa.field('libraryselection', pa.string()),
    pa.field('librarysource', pa.string()),
    pa.field('geo_loc_name_country_calc', pa.string()),
    pa.field('geo_loc_name_country_continent_calc', pa.string()),
    pa.field('mbytes', pa.int64()),
    pa.field('mbases', pa.int64()),
    pa.field('releasedate', pa.timestamp('ms')),
    pa.field('collection_date', pa.timestamp('ms')),
    pa.field('lat_lon_src', pa.string()),
    pa.field('lat_lon_raw', pa.string()),
    pa.field('latitude', pa.float64()),
    pa.field('longitude', pa.float64()),
    pa.field('latitude_precision', pa.float64()),
    pa.field('longitude_precision', pa.float64()),
    pa.field('has_latlon', pa.bool_()),
    pa.field('rg_country_code', pa.string()),
    pa.field('rg_city', pa.string()),
    pa.field('rg_country', pa.string()),
    pa.field('gc_continent', pa.string()),
    pa.field('gc_continent_code', pa.string())
    ])

    pd_output.to_parquet("geo-sra/geo-sra.parquet.gzip",filesystem=s3, schema = SCHEMA, compression='gzip')

    #pd_dict_k.to_parquet('/data/database/geo-sra/sra-metadata-attributes.parquet')

    print('#END - Write parquet')
    #############################################################################

    elapsed_time = time.time() - st

    print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

#
# Latitude and Longitude metadata
#

# geographic_location__latitude_and_longitude__sam *
# geographic_location__latitudeandlongitude__sam *
# geographic_location__latitude__sam *
# geographic_location__longitude__sam *

# geographiclocation_latitude__sam *

# latitude__and_longitude_sam *
# latitude_and_longitude_sam *

# latitude_deg_sam *
# latitude_end_sam
# longitude_deg_sam *
# longitude_end_sam

# latitude_dd_sam *
# longitude_dd_sam *

# latitude_sam *
# longitude_sam *

# lat_sam *
# lon_sam *

# latitude_start_sam
# longitude_start_sam

# lattitude_and_logitude_sam *
# longitude_lattitude_primer_1_sam

# lat_lon_run *
# lat_lon_sam *
# latlon_sam *
# lat_lon_dms_sam *
# lat_long_correct_sam *

# lat_lon_sam_s_dpl34 *
# lat_lon_sam_s_dpl1 *
# geographic_location__latitude__sam_s_dpl4 *
# geographic_location__longitude__sam_s_dpl5 *
# location_coordinates_sam *
# other_gps_coordinates_sam

#
# Geography
#

# isolation_source    
# geographic_location *

#
# Altitude
#

# altitude_masl_sam
# altitude_sam
# geographic_location__altitude_elevation__sam
# geographic_location__altitude__sam

#
# Depth
#

# avg_depth_cm__sam
# core_depth__m__sam
# depth_category_sam
# depth_end_sam
# depth_in_core_in_cm_run
# depth_in_core_in_cm_sam
# depth__mbsf__sam
# depth__meters__sam
# depth__m_sam
# depth__m__sam
# depth_m_sam
# depth_m__sam
# depth_sam
# depth_sample_sam
# depth_start_sam
# geographic_location__depth__sam
# sample_depth_m_sam
# sample_depth_or_location_sam
# sample_depth_sam
# sampling_depth_m_sam
# sampling_depth_sam
# secchi_depth_sam
# secchi_depth_visibility__m__sam
# sediment_depth_cm_sam
# sediment_depth_sam
# station_depth_sam
# stationdepth_sam
# total_depth__ft__sam
# total_depth_of_the_water_column_sam
# total_depth_of_water_column_sam
# tot_depth_water_col_m_sam
# tot_depth_water_col_sam
# water_depth__m__sam
# well_depth_sam

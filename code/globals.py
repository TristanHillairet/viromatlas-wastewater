### May 2024
### Tristan Hillairet
### Global script to store parameters and choose a script execution

####################################################################

################
# IMPORTATIONS #
################

import time
import numpy as np
import os

##############
# PARAMETERS #
##############

#
# done parameter (bool) status of the execution of a cell 
#

done = False

def get_done():
    return done

def set_done(bool):
    global done  
    done = bool

#
# Execution path of the globals.py script
#

exec_path = os.getcwd()

######################
# SCRIPTS EXECUTIONS #
######################

if __name__ == "__main__":

    tsta = time.time()

    script = str(input("\033[93mScript to execute (e to exit) : \033[0m"))

    try:

        #
        # Script exit
        #

        if script == "e":
            print("\033[93m#---Exit script execution\033[0m")

        #
        # Script extract_sra_latlon.py to parse geo datas
        #
            
        elif script == "extract_sra_latlon":
            with open(f'{exec_path}/code/geo_qual/extract_sra_latlon.py') as f:
                code = f.read()
                exec(code)
                tend = time.time()
                t  = np.round(tend - tsta,decimals=1)
                print(f"\033[93m#---Temps total d'exécution [{t}s]\033[0m")
        
        #
        # Script graph_lib.py helper
        # 

        elif script == "graph_lib":
            with open(f'{exec_path}/code/representations/graph_lib.py') as f:
                code = f.read()
                exec(code)
                tend = time.time()
                t  = np.round(tend - tsta,decimals=1)
                print(f"\033[93m#---Temps total d'exécution [{t}s]\033[0m")
        
        #
        # Handle script not found
        #
        
        else:
            print(f"\033[93m#---Script name '{script}' not found\033[0m")

    except Exception as e:
        print(e)



from pathlib import Path
import numpy as np
import random 
from datetime import datetime
import math
import pickle 
from gurobipy import *




def load_data(args):
    # Variables start with "h", "p", "r" represent: hcp, patient and room respectively
    # Load MICU data
    if args.data == "MICU":
        f_path = Path("data/data_MICU.npz")
        if not f_path.is_file():
            print("Cannot find MICU dataset!")
            return 
        data = np.load(f_path, allow_pickle = True)
        h_visits, p_stay, h_list, r_list, h_type, r_dists = data['hcp_visits'], \
                            data['patients_stay'], data['hcp_list'], data['room_list'], data['hcp_type'][()], data['room_to_room_distance'][()]
        return h_visits, p_stay, h_list, r_list, h_type, r_dists
    
    # Load LTCF small data
    elif args.data == "LTCF_small":
        f_path = Path("data/data_LTCF_small.npz")
        if not f_path.is_file():
            print("Cannot find LTCF small dataset!")
            return 
        data = np.load(f_path, allow_pickle = True)
        h_visits, h_list, r_list, h_type, r_type, r_dists = data['hcp_visits'], \
                            data['hcp_list'], data['room_list'], data['hcp_type'][()], data['room_type'][()], data['room_to_room_distance'][()]
        return h_visits, h_list, r_list, h_type, r_type, r_dists

    # Load LTCF large data
    elif args.data == "LTCF_large":
        f_path = Path("data/data_LTCF_large.npz")
        if not f_path.is_file():
            print("Cannot find LTCF_large dataset!")
            return 
        data = np.load(f_path, allow_pickle = True)
        h_visits, h_list, r_list, h_type, r_type, r_dists = data['hcp_visits'], \
                            data['hcp_list'], data['room_list'], data['hcp_type'][()], data['room_type'][()], data['room_to_room_distance'][()]
        return h_visits, h_list, r_list, h_type, r_type, r_dists



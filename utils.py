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


<<<<<<< HEAD
=======


# def find_MICU_baseline_cost(hcp_visits, hcp_list, room_list, room_distances, hcp_type):
#     room_demand_am = {}
#     room_demand_pm = {}
#     hcp_load = {}
#     hcp_mobility = {}
#     # Initialize all HCP's load, mobility and patient room demands
#     # Initial demands/loads/mobilities are zero
#     for hcp in hcp_list:
#         hcp_load[hcp] = 0
#         hcp_mobility[hcp] = 0

#     for room in room_list:
#         room_demand_am[room] = 0
#         room_demand_pm[room] = 0

#     for i in range(hcp_visits.shape[0]):
#         hcp, room, start_t, end_t = hcp_visits[i]
#         hcp_load[hcp]+= (end_t - start_t).total_seconds()/60.0 
#         if 'am' in hcp_type[hcp] or start_t.hour<12:
#             room_demand_am[room]+= (end_t - start_t).total_seconds()/60.0
#         else:
#             room_demand_pm[room]+= (end_t - start_t).total_seconds()/60.0
        

#         # For mobility find the previous room where the HCP has visited
#         idx = np.argwhere(hcp_visits[:i,0]==hcp)
#         if(len(idx)!=0):
#             last_idx = idx[-1,0]
#             hcp_mobility[hcp]+= room_distances[(hcp_visits[last_idx,1], room)]
#     return hcp_load, hcp_mobility, room_demand_am, room_demand_pm


# def find_LTCF_baseline_cost(hcp_visits, hcp_list, room_list, room_distances):
#     room_demand = {}
#     hcp_load = {}
#     hcp_mobility = {}
#     # Initialize all HCP's load, mobility and patient room demands
#     # Initial demands/loads/mobilities are zero
#     for hcp in hcp_list:
#         hcp_load[hcp] = 0
#         hcp_mobility[hcp] = 0

#     for room in room_list:
#         room_demand[room] = 0

#     for i in range(hcp_visits.shape[0]):
#         hcp, room, start_t, end_t = hcp_visits[i]
#         if type(start_t) in [float, int]:
#             hcp_load[hcp]+= (end_t - start_t)*20/60.0 # each time step ~ 20 seconds
#             room_demand[room]+= (end_t - start_t)*20/60.0
#         else:
#             hcp_load[hcp]+= (end_t - start_t).total_seconds()/60.0 
#             room_demand[room]+= (end_t - start_t).total_seconds()/60.0
        

#         # For mobility find the previous room where the HCP has visited
#         idx = np.argwhere(hcp_visits[:i,0]==hcp)
#         if(len(idx)!=0):
#             last_idx = idx[-1,0]
#             hcp_mobility[hcp]+= room_distances[(hcp_visits[last_idx,1], room)]
#     return hcp_load, hcp_mobility, room_demand


# def compute_probability(interval_list, trans_prob, source_room, dest_room):
#     # Probability of disease transmission from room u to room v
#     suffix_cnt = [ 0 for item in interval_list]
#     prefix_cnt = [ 0 for item in interval_list]
#     cnt = 0
#     for i in range(len(interval_list)-1, -1, -1):
#         suffix_cnt[i] = cnt
#         if interval_list[i] == dest_room:
#             cnt+=1
#     cnt = 0
#     for i in range(len(interval_list)):
#         prefix_cnt[i] = cnt
#         if interval_list[i] == source_room:
#             cnt+=1

#     prob = 0
#     for i in range(len(interval_list)):
#         if interval_list[i]==source_room:
#            prob+= ( (1-trans_prob)**prefix_cnt[i] ) * trans_prob * ( (1- (1-trans_prob)**suffix_cnt[i] ) )
#     return prob

# def compute_transmission_weight_between_rooms(args, room_list, hcp_list):
#     interval_length = 30 # seconds
#     trans_prob = 0.005
#     room_weight_map = {}
    
#     prob_dic = {}

#     # visit_history = {}
#     # for i in range(hcp_visits.shape[0]):
#     #     hcp, room, start_t, end_t = hcp_visits[i]
#     #     if (hcp,room) not in visit_history:
#     #         visit_history[(hcp, room)] = []
#     #     visit_history[(hcp, room)].append([start_t, end_t])
#     baseline_visit_graph = dump_load_pickle_object("load", filename="dumps/mobility/"+args.dataset+"/baseline_visit_graph")
#     for r_u in room_list:
#         for r_v in room_list:
#             if r_u == r_v:
#                 continue
#             for hcp in hcp_list:
#                 all_visits = []
#                 if (hcp, r_u) in baseline_visit_graph:
#                     for visit in baseline_visit_graph[(hcp, r_u)]:
#                         all_visits.append([visit[0], visit[1], "r_u"])

#                 if (hcp, r_v) in baseline_visit_graph:
#                     for visit in baseline_visit_graph[(hcp, r_v)]:
#                         all_visits.append([visit[0], visit[1], "r_v"])
#                 all_visits = sorted(all_visits, key=lambda x: x[0])

#                 # create a list of intervals (fixed length) from the ordered visits during shift s_i
#                 # if a visit duration is more than the size of fixed interval, then break it into different intervals
#                 # Handle fraction: if visit length is not evenly divisible by intervals then round it 
#                 interval_list = []
#                 for visit in all_visits:
#                     if type(visit[1]) in [int, float]:
#                         duration = (visit[1] - visit[0])
#                     else:
#                         duration = (visit[1] - visit[0]).total_seconds()
#                     for k in range(int(duration//interval_length)):
#                         # mark the interval with room marker
#                         # during this interval hcp h_i was in that marked room
#                         interval_list.append(visit[2])
#                         duration-=interval_length

#                     # Round here
#                     if duration>interval_length/2:
#                         interval_list.append(visit[2])

#                 prob_uv = compute_probability(interval_list = interval_list, trans_prob = trans_prob, source_room="r_u", dest_room = "r_v")
#                 prob_vu = compute_probability(interval_list = interval_list, trans_prob = trans_prob, source_room="r_v", dest_room = "r_u")

#                 prob_dic[(r_u, r_v, hcp)] = prob_uv
#                 prob_dic[(r_v, r_u, hcp)] = prob_vu

#     room_weight_map = {}
#     for room_u in room_list:
#         for room_v in room_list:
#             if room_u==room_v:
#                 continue
#             if (room_u, room_v) not in room_weight_map:
#                 room_weight_map[(room_u, room_v)] = 0
#             prob = 1
#             for hcp in hcp_list:
#                 if (room_u, room_v, hcp) in prob_dic:
#                     prob*= (1-prob_dic[( room_u, room_v, hcp)])
#             prob = 1 - prob
#             room_weight_map[(room_u,room_v)] += prob

#     # Integrate the weight also from the reverse room order
#     tmp = {}
#     for key in room_weight_map:
#         if (key[1],key[0]) in room_weight_map:
#             tmp[key]=room_weight_map[key] + room_weight_map[(key[1], key[0])]
#         else:
#             tmp[key]=room_weight_map[key]
    
#     # Make valid probability between 0 and 1
#     room_weight_map = tmp
#     for key in room_weight_map:
#         room_weight_map[key] = room_weight_map[key]/2

#     # Make transmission weight undirected
#     edge_weight = {}
#     for key in room_weight_map:
#         room_id_1 = key[0]
#         room_id_2 = key[1]
#         edge_weight[(room_id_1, room_id_2)] =  room_weight_map[key]
#         edge_weight[(room_id_2, room_id_1)] = room_weight_map[key]
    
#     return edge_weight

# #Dump/Load Pickle Object
# def dump_load_pickle_object(action="", filename="", data=""):
#     if action == "dump":
#         dump = open(filename, 'wb')
#         pickle.dump(data, dump)
#         dump.close()
#         return None
#     elif action == "load":
#         dump = open(filename, 'rb')
#         data = pickle.load(dump)
#         dump.close()
#         return data

>>>>>>> f6ea279e5df051401a6b26d0eba70c67b44dcfd6

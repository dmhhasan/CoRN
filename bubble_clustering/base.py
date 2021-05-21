from abc import ABC, abstractmethod
import pickle
from gurobipy import *
import random 
from datetime import datetime, timedelta
import numpy as np
import math
from pathlib import Path
import json


class BaseClustering(ABC):

    @abstractmethod
    def create_random_bubbles(self):
        pass 
    @abstractmethod
    def create_ILP_bubbles(self):
        pass
    @abstractmethod
    def find_baseline_load_demand_mobility(self):
        pass

    def calculate_transmission_weight(self, base_visit_graph):
        intrvl_len = 30 # seconds
        trans_prob = 0.005
        r_weight_map = {}

        def compute_probability(intrvl_list, trans_prob, src_room, dst_room):
            # Probability of disease transmission from room u to room v
            suf_cnt = [ 0 for i in intrvl_list]
            pre_cnt = [ 0 for i in intrvl_list]
            cnt = 0
            for i in range(len(intrvl_list)-1, -1, -1):
                suf_cnt[i] = cnt
                if intrvl_list[i] == dst_room:
                    cnt+=1
            cnt = 0
            for i in range(len(intrvl_list)):
                pre_cnt[i] = cnt
                if intrvl_list[i] == src_room:
                    cnt+=1

            prob = 0
            for i in range(len(intrvl_list)):
                if intrvl_list[i]==src_room:
                    prob+= ( (1-trans_prob)**pre_cnt[i] ) * trans_prob * ( (1- (1-trans_prob)**suf_cnt[i] ) )
            return prob

        def calculate_weight_for_individual_hcp(args, base_visit_graph):
            prob_dic = {}
            for r1 in self.r_list:
                for r2 in self.r_list:
                    if r1 == r2:
                        continue
                    for h in self.h_list:
                        all_visits = []
                        if (h, r1) in base_visit_graph:
                            for v in base_visit_graph[(h, r1)]:
                                all_visits.append([v[0], v[1], "r1"])

                        if (h, r2) in base_visit_graph:
                            for v in base_visit_graph[(h, r2)]:
                                all_visits.append([v[0], v[1], "r2"])
                        all_visits = sorted(all_visits, key=lambda x: x[0])

                        # create a list of intervals (fixed length) from the ordered visits during shift s_i
                        # if a visit duration is more than the size of fixed interval, then break it into different intervals
                        # Handle fraction: if visit length is not evenly divisible by intervals then round it 
                        intrvl_list = []
                        for v in all_visits:
                            if args.data == 'LTCF_small' or args.data == 'LTCF_large': # if args.data == 'LTCF_small' or 'LTCF_large'
                                t = (v[1] - v[0])
                            else:
                                t = (v[1] - v[0]).total_seconds()
                            for k in range(int(t//intrvl_len)):
                                # mark the interval with room marker
                                # during this interval hcp h_i was in that marked room
                                intrvl_list.append(v[2])
                                t-=intrvl_len

                            # Round here
                            if t>intrvl_len/2:
                                intrvl_list.append(v[2])

                        prob_uv = compute_probability(intrvl_list, trans_prob, "r1", "r2")
                        prob_vu = compute_probability(intrvl_list, trans_prob, "r2", "r1")

                        prob_dic[(r1, r2, h)] = prob_uv
                        prob_dic[(r2, r1, h)] = prob_vu
            return prob_dic
    
        prob_dic = calculate_weight_for_individual_hcp(self.args, base_visit_graph)
        
        # For MICU all rooms are patient rooms; 
        # Therefore, send the r_list for clustering
        rooms = self.r_list 
        # For LTCF there are also dining rooms and others 
        # Remove non-patient rooms before clustering
        if self.args.data == 'LTCF_small' or self.args.data == 'LTCF_large':
            rooms = [r for r in self.r_list if self.r_type[r]=='in-room']

        # Calculate weight between pair of rooms for all hcps
        for r1 in rooms:
            for r2 in rooms:
                if r1==r2:
                    continue
                if (r1, r2) not in r_weight_map:
                    r_weight_map[(r1, r2)] = 0
                prob = 1
                for h in self.h_list:
                    if (r1, r2, h) in prob_dic:
                        prob*= (1-prob_dic[( r1, r2, h)])
                prob = 1 - prob
                r_weight_map[(r1,r2)] += prob

        # Integrate the weight also from the reverse room order
        tmp = {}
        for key in r_weight_map:
            if (key[1],key[0]) in r_weight_map:
                tmp[key]=r_weight_map[key] + r_weight_map[(key[1], key[0])]
            else:
                tmp[key]=r_weight_map[key]
        
        # Make valid probability between 0 and 1
        r_weight_map = tmp
        for key in r_weight_map:
            r_weight_map[key] = r_weight_map[key]/2

        # Make transmission weight undirected for each edge
        e_weight = {}
        for key in r_weight_map:
            r1 = key[0]
            r2 = key[1]
            e_weight[(r1, r2)] =  r_weight_map[key]
            e_weight[(r2, r1)] = r_weight_map[key]
        
        for key in e_weight:
            e_weight[key] = int(e_weight[key]*1000)
    
        return e_weight



    def store_bubble_clustering(self):
        # Write a custom np_encoder for writing json with int64 type value
        # Otherwise, convert all writable items to python int
        def np_encoder(object):
            if isinstance(object, (np.generic, np.ndarray)):
                return object.item()

        # Save bubbles 
        dic = {}
        for key in self.clustering:
            dic["Bubble "+str(key)] = self.clustering[key]
        Path("output/clustering/"+self.args.data+"/").mkdir(parents=True, exist_ok=True)
        file = "output/clustering/"+self.args.data+"/ILP_"+str(self.n_bubble)+"_bubbles.json"
        with open(file, 'w') as f:
            json.dump(dic, f, sort_keys=True, indent=4, default=np_encoder)

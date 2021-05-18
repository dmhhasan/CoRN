
from abc import ABC, abstractmethod
from datetime import datetime, timedelta 
import random
import multiprocessing as mp
import numpy as np
import math
from pathlib import Path


class BaseSimulation(ABC):

    def __init__(self):
        #  W is the "wait time" or incubation period, in
        # days, and T is the temporal duration, also in days. An infection
        # occurs (early) on day 0; symptoms emerge on day W, and then the
        # infection lasts through day W+T.    
        self.W=5
        self.T=10

        # The first, the exp/exp model, shedding ramps up exponentially
        # (faster) from day 0 and peaks at day W, then exponentially ramps down
        # (slower) for T days through day W+T-1. Symptoms emerge at peak shedding.
        self.MODEL_TYPE_UNI_UNI = 'uni_uni'
        self.MODEL_TYPE_EXP_EXP = 'exp_exp'
        self.shedding_dic = {}
        self.shedding_dic['exp_exp'] = [0.001, 0.0039, 0.0156, 0.0625, 0.25, 1, 0.5, 0.25, 0.125, 0.0625, 0.0312, 0.0156, 0.0078, 0.0039, 0.001,0]
        self.shedding_dic['uni_uni'] = [0.01, 0.01, 0.01, 0.01, 0.01, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
        self.shedding_scale = .1
        self.model_type = self.MODEL_TYPE_EXP_EXP

        self.STATUS_INFECTED = 'infected'
        self.STATUS_SHEDDING = 'shedding'
        self.STATUS_QUARANTINE = 'qurantine'
        self.STATUS_RECOVERED = 'recovered'

        # External transmission 
        self.replicte_cnt_with_ext_leave = 0
        self.replicte_cnt_with_ext_reach = 0

    def initialize_replicate(self):
        # Record different cost values
        self.infection_list = {}

        self.r_demand = {}
        self.r_unmet_demand = {}
        for r in self.r_list:
            self.r_demand[r] = 0
            self.r_unmet_demand[r] = 0
        self.h_load = {}
        self.h_mobility = {}
        for h in self.h_list:
            self.h_load[h] = 0
            self.h_mobility[h] = 0
        
        # Initialize HCP availability
        self.h_availability = {}
        if  self.args.data == 'MICU':
            for h in self.h_list:
                self.h_availability[h] = datetime(1970, 1, 1)
        else:
            for h in self.h_list:
                self.h_availability[h] = 0
        
        # How many external transmission of two kinds occur
        self.ext_leave_cnt =0
        self.ext_reach_cnt = 0



    @abstractmethod
    def run_covid19_simulation(self):
        pass

    def update_infection_status(self):
        # Update infection status 
        for key in self.infection_list:
            if self.infection_list[key].inf_day==1:
                self.infection_list[key].status = self.STATUS_SHEDDING
            elif self.infection_list[key].inf_day == self.W+self.T:
                self.infection_list[key].status = self.STATUS_RECOVERED
    

    def find_overlapped_contact(self, cur_room, h_start, h_end):
        overlapped_contact_idx = []
        for i in range(len(self.h_visits)):
            hcp, room, start, end = self.h_visits[i]
            if (cur_room == room and ( h_start<=start<=h_end) ): # dont add  cur_start_time<=e<=cur_end_time to avoid duplicate
                overlapped_contact_idx.append(i)
        return overlapped_contact_idx

    def get_random_nurse(self, cur_room, hcws_per_room, idx_start, idx_hcp):
        filter_list = []
        for h in  hcws_per_room[cur_room][self.h_type[idx_hcp]]:
            if self.h_availability[h]<idx_start:
                filter_list.append(h)
        return filter_list[random.randint(0,len(filter_list)-1)] if len(filter_list)>0 else None
    
    def find_overlapping_period(self, startA, endA, startB, endB):
        latest_start = max(startA, startB)
        earliest_end = min(endA, endB)
        if self.args.data == 'MICU':
            delta = (earliest_end - latest_start).total_seconds()
        else:
            delta = (earliest_end - latest_start)
        overlap = max(0, delta)
        return overlap

    def get_long_exposure_prob(self, cur_start_time, cur_end_time, trans_prob):
        if self.args.data == 'MICU':
            visit_duration = (cur_end_time - cur_start_time).total_seconds()
        else:
            visit_duration = (cur_end_time - cur_start_time)
        long_exposure_trans_prob = 1 - math.pow((1 - trans_prob), math.ceil(visit_duration / 30)) # 30 seconds
        return long_exposure_trans_prob

    def store_rewired_network(self, adj, node_list): 
        isolated_list = list(np.where(~adj.any(axis=1))[0])
        Path("output/rewired_network/"+self.args.data+"/"+str(self.args.simulation)+"/").mkdir(parents=True, exist_ok=True)
        node_arr = np.array(node_list)
        np.savez("output/rewired_network/"+self.args.data+"/"+str(self.args.simulation)+"/"+"bubble_"+str(self.args.n_bubble)+"_adjacency.npz", name1=adj, name2=node_arr)

    def process_simulation_results(self, results):

        # Initialize with 0's
        r0_list = []
        inf_cnt_list = []
        h_load_sum = {}
        h_mobility_sum = {}
        for h in self.h_list:
            h_load_sum[h] = 0
            h_mobility_sum[h] = 0
        r_demand_sum = {}
        r_unmet_demand_sum = {}
        for r in self.r_list:
            r_demand_sum[r] = 0
            r_unmet_demand_sum[r] = 0

        # Run for each replicate and add each quantity
        for i in range(self.args.n_replicate):
            inf_list, h_load, r_demand, r_unmet_demand, h_mobility, ext_leave_cnt, ext_reach_cnt = results[i]

            # R0 of current replicate based on all HCPs
            list = []
            for key in inf_list:
                inf = inf_list[key]
                list.append(len(inf.secondary_infs))
            r0 = np.mean(list)
            r0_list.append(r0)
            inf_cnt_list.append(len(inf_list))
            # update hcp load
            for h in h_load:
                h_load_sum[h]+=h_load[h]
            # update hcp mobility
            for h in h_mobility:
                h_mobility_sum[h]+=h_mobility[h]
            # update room demand
            for r in r_demand:
                r_demand_sum[r]+=r_demand[r]
            # update unmet room demand
            for r in r_unmet_demand:
                r_unmet_demand_sum[r]+=r_unmet_demand[r]
            # update ext leave/reach replicate
            if ext_leave_cnt > 0:
                self.replicte_cnt_with_ext_leave+=1
            if ext_reach_cnt > 0:
                self.replicte_cnt_with_ext_reach+=1
        
        # calculate average of the sum for each quantity
        r0_avg = np.mean(r0_list)
        h_load_list =[h_load_sum[h]/self.args.n_replicate for h in h_load_sum]
        h_mobility_list =[h_mobility_sum[h]/self.args.n_replicate for h in h_mobility_sum]
        r_demand_list = [r_demand_sum[r]/self.args.n_replicate for r in r_demand_sum]
        r_unmet_demand_list =[r_unmet_demand_sum[r]/self.args.n_replicate for r in r_unmet_demand_sum]
        ext_leave_cnt_fraction = self.replicte_cnt_with_ext_leave / self.args.n_replicate
        ext_reach_cnt_fraction = self.replicte_cnt_with_ext_reach / self.args.n_replicate

        # Convert to per day in hour
        summary = {"Infection count": np.mean(inf_cnt_list), "R0": r0_avg, "Avg. HCP load": np.mean(h_load_list)/(30*60), "Avg. HCP mobility": np.mean(h_mobility_list)/30, "Avg. room demand": np.mean(r_demand_list)/(30*60),
                     "Avg. unmet demand": np.mean(r_unmet_demand_list)/(30*60), "Fraction of external leave":ext_leave_cnt_fraction, "Fraction of external reach:":ext_reach_cnt_fraction}
        
        return summary
        
class SimulationData:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class InfectionData:
    def __init__(self, inf_day, is_first_infected, infected_by, trans_prob, infected_at, status):
        self.inf_day = inf_day
        self.is_first_infected = is_first_infected
        self.secondary_infs = []
        self.infected_by = infected_by
        self.trans_prob = trans_prob
        self.infected_at = infected_at
        self.status = status
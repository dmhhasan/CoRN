from .base import *


class LTCFCovid19Simulation(BaseSimulation):
    def __init__(self, args, h_visits, h_list, r_list, h_type, r_type, r_dists):
        super().__init__()
        self.args = args 
        self.h_list = h_list
        self.r_list = list(r_list) 
        self.h_type = h_type
        self.r_type = r_type
        self.nrs_list = [h for h in self.h_list if self.h_type[h]=='nurse']
        self.non_nrs_list = [h for h in self.h_list if self.h_type[h]=='non-nurse']
        self.h_visits = h_visits
        self.r_dists = r_dists
        if self.args.data == "LTCF_small":
            self.random_mixing_prob= 0.018057 # This value is calculated based on outdoor contacts
        elif self.args.data == "LTCF_large":
            self.random_mixing_prob= 0.052412
    
    def run_covid19_simulation(self):
        random.seed(42)
        inf_srcs = random.choices(self.nrs_list, k=self.args.n_replicate)
        self.param = []
        for i in range(self.args.n_replicate):
            if self.args.simulation == 'base':
                clustering = None 
            else:
                clustering = self.args.clustering[i]
            s_data = SimulationData(
                                    clustering = clustering,
                                    inf_src = inf_srcs[i],
                                    rep_id = i,
                                    trans_prob = self.args.trans_prob
                                    )
            self.param.append(s_data)

        pool = mp.Pool(self.args.n_cpu)
        results = pool.map(self.covid19_simulation, self.param)
        pool.close()
        pool.join()
        return self.process_simulation_results(results)
    

    def covid19_simulation(self, data):
        random.seed(datetime.now())
        self.initialize_replicate()
        # Save an adjacency matrix for rewired contact network just once
        if data.rep_id == 1:
            if self.args.simulation == 'rewired_ilp' or self.args.simulation == 'rewired_random':
                node_list = self.non_nrs_list+self.nrs_list+self.r_list
                node_mapping = {}
                index_mapping = {}
                for n in node_list:
                    index_mapping[len(node_mapping)] = n
                    node_mapping[n] = len(node_mapping)
                adj = np.zeros((len(node_list),len(node_list)))


        # Infect the source and add to list
        rand_hcp = data.inf_src
        infectionData = InfectionData(inf_day=0,is_first_infected = True, infected_by=None, trans_prob=data.trans_prob, infected_at=None, status=self.STATUS_SHEDDING)
        self.infection_list[rand_hcp] = infectionData
    
        # For rewired simulation initialize the source bubble
        if self.args.simulation == 'rewired_ilp' or self.args.simulation=='rewired_random':
            src_bubble = data.clustering['h_bubble'][rand_hcp]
        else:
            src_bubble = None

        # Keep last visits record
        last_visit = {}
        initial_time = 0
        for h in self.h_list:
            last_visit[h] = {'start':initial_time, 'end':initial_time, 'room': None}


        for day in range(1,31):
            random.seed(datetime.now())

            # Update infection status 
            self.update_infection_status()

            for h in self.h_list:
                self.h_availability[h] = 0

            # Keep track of HCP_Patient interaction index that are processed together 
            # for overlapping visits in the same room at the same time
            overlapped_processed_contact = []

            for i in range(len(self.h_visits)):
                if i in overlapped_processed_contact:
                    continue
                cur_hcp, cur_room, h_start, h_end = self.h_visits[i]
                dining_interaction_dic = {}
                # If a nurse visits dining room, just find random nurse from the same bubble and replace
                # add the nurse for the dummy dining room of the source bubble
                # then continue
                if self.r_type[cur_room]=='dining' and self.h_type[cur_hcp]=='nurse':
                    if self.args.simulation == 'rewired_random' or self.args.simulation=='rewired_ilp':
                        random_h = cur_hcp # send cur_hcp to dining 
                        if random_h is not None:
                            self.h_availability[random_h] = h_end
                        bubble = data.clustering['h_bubble'][random_h]
                        if bubble not in dining_interaction_dic:
                            dining_interaction_dic[bubble] = []
                        dining_interaction_dic[bubble].append([random_h, cur_room, h_start, h_end])
                    else:
                        self.h_availability[cur_hcp] = h_end
                        if cur_room not in dining_interaction_dic:
                            dining_interaction_dic[cur_room] = []
                        dining_interaction_dic[cur_room].append([cur_hcp, cur_room, h_start, h_end])
                    continue
                elif self.r_type[cur_room] != 'in-room':
                    continue

                patient_id = cur_room
                overlapped_contact_idx = self.find_overlapped_contact(cur_room, h_start, h_end)
                
                # In perturb simulation, HCP of an index can be randomly replaced
                # Therefore, keep a mapping which index maps to which HCP
                rewired_hcp_mapping = {}
                
                for idx in overlapped_contact_idx:
                    overlapped_processed_contact.append(idx)
                    rewired_hcp_mapping = self.do_rewiring(idx, cur_room, data, rewired_hcp_mapping)
                    idx_hcp, idx_room, idx_start, idx_end = self.h_visits[idx]
                    # mapped_hcp could be either same or different from index_hcp
                    # depending on if perturb simulation and nurse type
                    mapped_hcp = rewired_hcp_mapping[idx]
                    # mapped_hcp is None only when demand is unmet.
                    # don't need update unmet demand here as it is already done above
                    if mapped_hcp is None:
                        continue

                    # Save contact into adjacency matrix just once
                    if data.rep_id == 1:
                        if self.args.simulation == 'rewired_random' or self.args.simulation == 'rewired_ilp':
                            adj[node_mapping[mapped_hcp],node_mapping[cur_room]]+= idx_end - idx_start
                            adj[node_mapping[cur_room],node_mapping[mapped_hcp]]+= idx_end - idx_start

                    # update load and met demand here
                    val = (idx_end - idx_start)/60.0
                    self.h_load[mapped_hcp]+=val 
                    self.r_demand[cur_room]+=val 
                    mobility = self.r_dists[(last_visit[mapped_hcp]['room'], cur_room)] if last_visit[mapped_hcp]['room'] is not None else 0
                    self.h_mobility[mapped_hcp]+=mobility

                    # calculate environmental contamination if not infected
                    if mapped_hcp not in self.infection_list:
                        h_bubble = None if self.args.simulation=='base' else data.clustering['h_bubble']
                        env_prob = self.get_env_contamination(mapped_hcp, idx_start, last_visit, h_bubble)

                        if self.h_type[mapped_hcp]=='nurse' and random.random() < env_prob:
                            # HCP get infected from environmental exposure
                            infectionData = InfectionData(inf_day=0,is_first_infected = False, infected_by="env", trans_prob=data.trans_prob, infected_at="env", status=self.STATUS_INFECTED)
                            self.infection_list[mapped_hcp] = infectionData
                                
                        elif patient_id in self.infection_list and self.infection_list[patient_id].status == self.STATUS_SHEDDING:#and random.random() < long_exposure_trans_prob:
                            shedding = self.shedding_dic[self.model_type][self.infection_list[patient_id].inf_day]/self.shedding_scale
                            shedding*=data.trans_prob
                            scaled_shedding = self.get_long_exposure_prob(idx_start, idx_end, shedding)
                            if random.random() < scaled_shedding:
                                # HCP get infected by a patient
                                infectionData = InfectionData(inf_day=0,is_first_infected = False, infected_by="patient", trans_prob=data.trans_prob, infected_at=cur_room, status=self.STATUS_INFECTED)
                                self.infection_list[mapped_hcp] = infectionData
                                self.infection_list[patient_id].secondary_infs.append(mapped_hcp)
                                # if self.args.simulation != 'base':
                                #     self.update_ext_transmission_cnt(cur_room, data, src_bubble, mapped_hcp)
                    
                        
                # find the probability of patient being infected
                # by overlapped HCPs in the room 
                if patient_id not in self.infection_list:
                    patient_trans_prob = 1
                    infected_by = [] # which HCPs infect the patient and for how long it contacts
                    for idx in overlapped_contact_idx:
                        idx_hcp, idx_room, idx_start, idx_end = self.h_visits[idx]
                        mapped_hcp = rewired_hcp_mapping[idx]
                        if mapped_hcp == None:
                            continue
                        if mapped_hcp in self.infection_list and self.infection_list[mapped_hcp].status == self.STATUS_SHEDDING:
                            shedding = self.shedding_dic[self.model_type][self.infection_list[mapped_hcp].inf_day]/self.shedding_scale
                            shedding*=data.trans_prob
                            scaled_shedding = self.get_long_exposure_prob(idx_start, idx_end, shedding)
                            patient_trans_prob*=(1 - scaled_shedding)
                            infected_by.append([mapped_hcp, [idx_start, idx_end] ])

                    patient_trans_prob = (1-patient_trans_prob)
                    if random.random() < patient_trans_prob:
                        # Patient get infected by HCP
                        infectionData = InfectionData(inf_day=0,is_first_infected = False, infected_by="hcp", trans_prob=data.trans_prob, infected_at = cur_room, status=self.STATUS_INFECTED)
                        self.infection_list[patient_id] = infectionData
                        for h,ts in infected_by:
                            self.infection_list[h].secondary_infs.append(patient_id)
                        # if self.args.simulation != 'base':
                        #     if cur_room not in data.clustering['assignment'][src_bubble]['room']:
                        #         self.ext_leave_cnt+=1
                        #         self.ext_reach_cnt+=1

                # Calculate HCP-HCP transmission inside of the patient room
                hcp_hcp_inf_list = self.calculate_hcp_hcp_transmission(overlapped_contact_idx, rewired_hcp_mapping, data, cur_room, src_bubble)
                
                for key in hcp_hcp_inf_list:
                    self.infection_list[key] = hcp_hcp_inf_list[key]
                # Calculate HCP-HCP transmission inside of the shared place e.g., dining room
                hcp_hcp_dining_inf_list = self.calculate_dining_place_transmission(data, dining_interaction_dic)
                
                for key in hcp_hcp_dining_inf_list:
                    self.infection_list[key] = hcp_hcp_dining_inf_list[key]
                # update the last visits of overlapped HCPs
                for idx in overlapped_contact_idx:
                    mapped_hcp = rewired_hcp_mapping[idx]
                    if mapped_hcp == None:
                        continue
                    hcp, room, start, end = self.h_visits[idx]
      
                    last_visit[mapped_hcp] = {'start':start, 'end': end, 'room':cur_room} 

            for key in self.infection_list.keys():
                self.infection_list[key].inf_day+=1

        # Save rewired contact network just once
        if data.rep_id == 1:
            if self.args.simulation != 'base':
                self.store_rewired_network(adj, node_list)
        R0 = len(self.infection_list[data.inf_src].secondary_infs)

        if self.args.simulation != 'base':
            for key in self.infection_list:
                # Any non-nurse hcp infection means the pathogen leaves the bubble
                if key in self.h_list and 'nurse' not in self.h_type[key]:
                    self.ext_leave = True
                    # where an infected trasnmission happened
                    location = self.infection_list[key].infected_at 
                    if location in self.r_list and location not in data.clustering['assignment'][src_bubble]['room']:
                        self.ext_reach = True

        return self.infection_list, self.h_load, self.r_demand, self.r_unmet_demand, self.h_mobility, self.ext_leave, self.ext_reach, R0


    def get_env_contamination(self, cur_hcp, cur_time, last_visit, hcp_bubble):
        outside_start = last_visit[cur_hcp]['end'] # last_visit should not be the current visit of cur_hcp 
        outside_end = cur_time

        # Calculate overlapping between cur_hcp and
        # other cur shift HCP's during the outside room time period
        
        not_mixing_trans_prob = 1
        for h in self.h_list:
            # if the outside hcp not contagious 
            # chance of transmission is zero
            if h==cur_hcp or h not in self.infection_list:
                continue
            if h in self.infection_list and self.infection_list[h].status!=self.STATUS_SHEDDING:
                continue
            start = last_visit[h]['end']
            end = last_visit[h]['end']

            if outside_end-outside_start>60*60:
                outside_start = outside_end

            #overlapping = max(0, (start-outside_start).total_seconds()) + max(0, min((outside_end-end).total_seconds(), (outside_end-outside_start).total_seconds()))
            overlapping = self.find_overlapping_period(startA=start, endA=end, startB=outside_start, endB=outside_end)
            if overlapping!=0:
                prob = overlapping/(outside_end-outside_start)
            else:
                prob = 0
            if hcp_bubble is not None and h in hcp_bubble and cur_hcp in hcp_bubble and hcp_bubble[h] == hcp_bubble[cur_hcp]:
                #mixing_trans_prob = LTCF_INTRA_BUBBLE_MIXING_TRANS_PROB*prob
                mixing_trans_prob = self.random_mixing_prob*prob
            else:
                mixing_trans_prob = self.random_mixing_prob* 0.75*prob
            long_mixing_trans_prob = mixing_trans_prob#1 - math.pow((1 - mixing_trans_prob), math.ceil(overlapping / 30)) # 30 seconds

            not_mixing_trans_prob *=(1-long_mixing_trans_prob)
        env_contamination = 1-not_mixing_trans_prob
        return env_contamination



    def do_rewiring(self, idx, cur_room, data, rewired_hcp_mapping):
        idx_hcp, idx_room, idx_start, idx_end = self.h_visits[idx]
 
        if self.args.simulation == 'rewired_random' or self.args.simulation=='rewired_ilp':
            # perturb only nurses
            if self.h_type[idx_hcp]=='nurse':
                new_hcp = self.get_random_nurse(cur_room,data.clustering['h_per_room'], idx_start, idx_hcp)
                if new_hcp is not None:
                    rewired_hcp_mapping[idx] = new_hcp
                    self.h_availability[new_hcp] = idx_end
                else:
                    rewired_hcp_mapping[idx] = None
                    unmet_demand = (idx_end - idx_start)/60.0 
                    self.r_unmet_demand[idx_room]+=unmet_demand
            # For non-nurses do not change the HCP
            else:
                rewired_hcp_mapping[idx] = idx_hcp
        # For baseline do not change the HCP as well
        else:
            rewired_hcp_mapping[idx] = idx_hcp
        return rewired_hcp_mapping


    
    def calculate_hcp_hcp_transmission(self, overlapped_contact_idx, rewired_hcp_mapping, data, cur_room, src_bubble):
        # Here calculate the HCP-HCP transmission inside of the room
        # Find who are uninfected and what is the transmission probability
        hcp_hcp_infection_list = {}
        
        for idx_1 in overlapped_contact_idx:
            hcp1, room1, start1, end1 = self.h_visits[idx_1]
            mapped_hcp1 = rewired_hcp_mapping[idx_1]
            if mapped_hcp1 == None:
                continue
            mapped_hcp1_trans_prob = 1
            infected_by = []

            for idx_2 in overlapped_contact_idx:
                hcp2, room2, start2, end2 = self.h_visits[idx_2]
                mapped_hcp2 = rewired_hcp_mapping[idx_2]
                if mapped_hcp2 == mapped_hcp1 or mapped_hcp2 == None:
                    continue

                # Overlapped time is the contact duration
                # Overlapped time = (latest_start, earliest_end)
                ts_start = max(start1, start2)
                ts_end = min(end1, end2)

                if mapped_hcp2 not in self.infection_list or (mapped_hcp2 in self.infection_list and self.infection_list[mapped_hcp2].status!=self.STATUS_SHEDDING):
                        continue

                # who infects and for how long it contacts
                # find overlapping time between mapped_hcp1 and mapped_hcp2
                # calculate prob based on the overlapped time
                # overlapped_time in seconds
                overlapped_time = self.find_overlapping_period(start1, end1, start2, end2)
                shedding = self.shedding_dic[self.model_type][self.infection_list[mapped_hcp2].inf_day]/self.shedding_scale
                shedding*=data.trans_prob
                scaled_shedding = 1 - math.pow((1 - shedding), math.ceil(overlapped_time / 30)) # 30 seconds
                mapped_hcp1_trans_prob*=(1-scaled_shedding)
                infected_by.append([mapped_hcp2, [ts_start, ts_end] ])

            mapped_hcp1_trans_prob = (1-mapped_hcp1_trans_prob)
            if mapped_hcp1 not in self.infection_list and random.random() < mapped_hcp1_trans_prob:
                # HCP get infected from HCPs
                infectionData = InfectionData(inf_day=0,is_first_infected = False, infected_by="hcp", trans_prob=data.trans_prob, infected_at=cur_room, status=self.STATUS_INFECTED)
                    
                #infection_list[mapped_hcp1] = infectionData
                for h, contact_timestamp in infected_by:
                    self.infection_list[h].secondary_infs.append(mapped_hcp1)
                    
                hcp_hcp_infection_list[mapped_hcp1] = infectionData
                # if self.args.simulation != 'base':
                #     self.update_ext_transmission_cnt(cur_room, data, src_bubble, mapped_hcp1)

        return hcp_hcp_infection_list

    

    def calculate_dining_place_transmission(self, data, dining_interaction_dic):
        tmp_infection_list = {}
        for bubble in dining_interaction_dic:
            dining_hcps = dining_interaction_dic[bubble]
            prob = 1
            infected_by = []
            for hcp, room, start, end in dining_hcps:
                if hcp in self.infection_list and self.infection_list[hcp].status == self.STATUS_SHEDDING:
                    shedding = self.shedding_dic[self.model_type][self.infection_list[hcp].inf_day]/self.shedding_scale
                    shedding*=data.trans_prob
                    scaled_shedding = self.get_long_exposure_prob(start, end, shedding)
                    prob*=(1-scaled_shedding)
                    infected_by.append([hcp, [start, end] ])
            prob = 1 - prob
            for hcp, room, start, end in dining_hcps:
                if hcp not in self.infection_list and random.random()<prob:
                    infectionData = InfectionData(inf_day=0,is_first_infected = False, infected_by="hcp", trans_prob=data.trans_prob, infected_at=room, status=self.STATUS_INFECTED)
                    for h, contact_timestamp in infected_by:
                        self.infection_list[h].secondary_infs.append(hcp)
                    tmp_infection_list[hcp] = infectionData
        return tmp_infection_list




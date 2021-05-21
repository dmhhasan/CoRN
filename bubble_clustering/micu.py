from .base import *

class MICUBubbleClustering(BaseClustering):
    def __init__(self, args, h_list, r_list, h_type, r_dist, h_visits, p_stay):
        #super().__init__(self, args, r_list, h_list)
        self.args = args 
        self.h_list = h_list  # HCP list
        self.r_list = r_list  # Room list
        self.h_type = h_type  # HCP type e.g. am_nurse, pm_nurse, non_nurse
        self.r_dist = r_dist  # Room to room distances in hops (1 hop ~ 3 meter)
        self.h_visits = h_visits
        self.p_stay = p_stay
        self.am_nrs_list = [h for h in self.h_list if self.h_type[h]=='am_nurse']
        self.pm_nrs_list = [h for h in self.h_list if self.h_type[h]=='pm_nurse']
        self.n_bubble = args.n_bubble
        self.bbl_list = np.arange(self.n_bubble)
        self.interaction_date = datetime(2011, 6, 3)

    def find_baseline_load_demand_mobility(self):
        self.base_r_demand_am = {} # Patient room demand during the morning hours
        self.base_r_demand_pm = {} # Patient room demand during the evenning hours
        self.base_hcp_load = {}    # HCP loads incurred by the patients
        self.base_h_mobility = {}  # Walking distance by each HCP

        # Repeat one day HCP-Room interactions over 30 days patients occupancy to create a new visit graph of 30 days; 
        # needed for computing transmission weights between rooms
        self.base_visit_graph = {} 


        # Initialize all HCP's load, mobility and patient room demands
        for h in self.h_list:
            self.base_hcp_load[h] = 0
            self.base_h_mobility[h] = 0
        for r in self.r_list:
            self.base_r_demand_am[r] = 0
            self.base_r_demand_pm[r] = 0

        last_visit = {}
        initial_time = self.h_visits[0][2]
        normalized_init_time = (initial_time - timedelta(minutes=(self.interaction_date.day - 1) * 1440) + timedelta(minutes= 1440))
        for h in self.h_list:
            last_visit[h] = {'start':normalized_init_time, 'end':normalized_init_time, 'room': None}

        for i in range(len(self.h_visits)):
            cur_hcp, cur_room, h_start, h_end = self.h_visits[i]

            # normalised_h_start = (h_start - timedelta(minutes=(self.interaction_date.day - 1) * 1440) + timedelta(minutes= day*1440))
            # normalised_h_end = (h_end - timedelta(minutes=(self.interaction_date.day - 1) * 1440) + timedelta(minutes= day*1440))

            # patient_id = None
            # for j in range(len(self.p_stay)):
            #     patient, room, p_start, p_end = self.p_stay[j]
            #     if room == cur_room and p_start <= normalised_h_start <= p_end and p_start <= normalised_h_end <= p_end:
            #         patient_id = patient
            #         break
            
            # Update visit graph (only for non-substituble HCPs i.e. HCP who do not belong to any bubble)
            if 'nurse' not in self.h_type[cur_hcp]:
                key = (cur_hcp, cur_room) 
                if key not in self.base_visit_graph:
                    self.base_visit_graph[key] = []
                self.base_visit_graph[key].append([h_start, h_end])

            # Update load, demand and mobility only for nurses
            #if patient_id!=None and 'nurse' in self.h_type[cur_hcp]:
                # update load and met demand here
            val = (h_end - h_start).total_seconds()/60.0
            self.base_hcp_load[cur_hcp]+= val
            if 'am' in self.h_type[cur_hcp]:
                self.base_r_demand_am[cur_room]+=val
            elif 'pm' in self.h_type[cur_hcp]:
                self.base_r_demand_pm[cur_room]+=val
            elif h_start.hour<12:
                self.base_r_demand_am[cur_room]+=val
            elif h_start.hour>=12:
                self.base_r_demand_pm[cur_room]+=val
            self.base_h_mobility[cur_hcp] += self.r_dist[(last_visit[cur_hcp]['room'], cur_room)] if last_visit[cur_hcp]['room'] is not None else 0
            
            # Keep track of the last visit
            last_visit[cur_hcp] = {'start':h_start, 'end':h_end, 'room':cur_room}
        return 0 


    # Overriding abstract method 
    def create_random_bubbles(self):
        # Use seed to generate different clustering for different experiments
        random.seed(datetime.now())

        # Random permutation
        random.shuffle(self.am_nrs_list)
        random.shuffle(self.pm_nrs_list)
        random.shuffle(self.r_list)
        random.shuffle(self.bbl_list)
        
        am_idx = 0
        pm_idx = 0
        r_idx = 0
        bbl_idx = 0
        
        # Hold information of which hcp belongs to which bubble
        self.h_bubble = {}
        self.h_per_room = {}
        self.clustering = {}
        for b in self.bbl_list:
            self.clustering[b]={'am_nurse':[], 'pm_nurse':[], 'room':[]}

        

        while True:
            b = self.bbl_list[bbl_idx]
            if am_idx < len(self.am_nrs_list):
                self.clustering[b]['am_nurse'].append(self.am_nrs_list[am_idx])
                self.h_bubble[self.am_nrs_list[am_idx]] = b
                am_idx+=1
            if pm_idx < len(self.pm_nrs_list):
                self.clustering[b]['pm_nurse'].append(self.pm_nrs_list[pm_idx])
                self.h_bubble[self.pm_nrs_list[pm_idx]] = b
                pm_idx+=1
            if r_idx < len(self.r_list):
                self.clustering[b]['room'].append(self.r_list[r_idx])
                r_idx+=1
            bbl_idx +=1
            if bbl_idx==len(self.bbl_list):
                bbl_idx=0
            if am_idx==len(self.am_nrs_list) and pm_idx==len(self.pm_nrs_list) and r_idx==len(self.r_list):
                break
        self.clustered_rooms = {}
        for key in self.clustering:
            self.clustered_rooms[key] = self.clustering[key]['room']

        for key in self.clustering.keys():
            rooms = self.clustering[key]['room']
            am_nrs = self.clustering[key]['am_nurse']
            pm_nrs = self.clustering[key]['pm_nurse']
            for r in rooms:
                if r not in self.h_per_room.keys():
                    self.h_per_room[r] = {"am_nurse":[], "pm_nurse":[]}
                for n in am_nrs:
                    self.h_per_room[r]["am_nurse"].append(n)
                for n in pm_nrs:
                    self.h_per_room[r]["pm_nurse"].append(n)

        return self.h_bubble, self.h_per_room, self.clustering


    # Overriding absract method
    def create_ILP_bubbles(self):
        self.find_baseline_load_demand_mobility()
        self.weights =  self.calculate_transmission_weight(self.base_visit_graph)
        # Add a certain value to each edge to avoid zero-weight edges 
        # Otherwise, the ILP solver misinterprets the problem space
        for key in self.weights:
            self.weights[key]+=1     

        self.h_bubble = {}
        self.h_per_room = {}
        self.clustering = {}

        if self.solve_integer_linear_program():
            for b in self.bbl_list:
                self.clustering[b]={'am_nurse':[], 'pm_nurse':[], 'room':[]}

            for b in self.bbl_list:
                for n in self.clustered_am_nrs[b]:
                    self.h_bubble[n] = b
                for n in self.clustered_pm_nrs[b]:
                    self.h_bubble[n] = b

            for b in self.bbl_list:
                self.clustering[b]['am_nurse'] = self.clustered_am_nrs[b]
                self.clustering[b]['pm_nurse'] = self.clustered_pm_nrs[b]
            
            for b in self.bbl_list:
                self.clustering[b]['room'] = self.clustered_rooms[b]

            for key in self.clustering.keys():
                rooms = self.clustering[key]['room']
                am_nurses = self.clustering[key]['am_nurse']
                pm_nurses = self.clustering[key]['pm_nurse']
                for r in rooms:
                    if r not in self.h_per_room.keys():
                        self.h_per_room[r] = {"am_nurse":[], "pm_nurse":[]}
                    for n in am_nurses:
                        self.h_per_room[r]["am_nurse"].append(n)
                    for n in pm_nurses:
                        self.h_per_room[r]["pm_nurse"].append(n)
            self.store_bubble_clustering()
        return self.h_bubble, self.h_per_room, self.clustering



    def solve_integer_linear_program(self):
        # Solve ILP problem for graph partitioning
        # Find num_bubbles partitions such that edge-cut is minimized in terms of edge weights

        # First create edge variables for all edges \in G
        # e_{uv} \in {0,1}; e_{uv}=1 if it is a cut edge otherwise e_{uv}=0
        E = []
        for key in self.weights:
            E.append((key[0],key[1]))
        # Create a list of edge weight associated with each edge e
        W = {}
        for key in self.weights:
            W[(key[0], key[1])] = self.weights[key]
        
        # Create another list of variables for each of the vertex and bubble
        nodes = set()
        for key in self.weights:
            nodes.add(key[0])
            nodes.add(key[1])
        self.V = list(nodes)

        C = {}
        for node in self.V:
            C[node] = 1

        L_max = math.ceil(len(self.V)/self.n_bubble)
        L_min = math.floor(len(self.V)/self.n_bubble)
        self.K = range(1, self.n_bubble+1)
        
        L_max_am = math.ceil(len(self.am_nrs_list)/self.n_bubble)
        L_min_am = math.floor(len(self.am_nrs_list)/self.n_bubble)
        L_max_pm = math.ceil(len(self.pm_nrs_list)/self.n_bubble)
        L_min_pm = math.floor(len(self.pm_nrs_list)/self.n_bubble)
        self.N_am = self.am_nrs_list
        self.N_pm = self.pm_nrs_list
        N_all = self.N_am + self.N_pm

        mdl = Model('CoRN')
        self.e= mdl.addVars(E, vtype=GRB.BINARY)
        self.x= mdl.addVars(self.V,self.K, vtype=GRB.BINARY)
        d= mdl.addVar(vtype=GRB.CONTINUOUS)
        # For am nurses
        self.z_am = mdl.addVars(self.K,N_all, vtype=GRB.BINARY)
        y = mdl.addVar(vtype=GRB.INTEGER)
        # For pm nurses
        self.z_pm = mdl.addVars(self.K,N_all, vtype=GRB.BINARY)
        
        mdl.modelSense = GRB.MINIMIZE

        mdl.setObjective(quicksum(self.e[u,v]*W[u,v] for u,v in E))
        mdl.addConstrs(self.e[u,v]>=(self.x[u,k]-self.x[v,k]) for k in self.K for u,v in E)
        mdl.addConstrs(self.e[u,v]>=(self.x[v,k]-self.x[u,k]) for k in self.K for u,v in E)
        mdl.addConstrs(self.r_dist[(u,v)]*(1-self.e[u,v])<=d for u,v in E)
        mdl.addConstrs(quicksum(self.x[v,k]*C[v] for v in self.V)<=L_max for k in self.K)
        mdl.addConstrs(quicksum(self.x[v,k]*C[v] for v in self.V)>=L_min for k in self.K)
        mdl.addConstrs(quicksum(self.x[v,k] for k in self.K)==1 for v in self.V)
        mdl.addLConstr(d<=self.args.D)

        # for am nurses
        mdl.addConstrs(quicksum(self.base_hcp_load[w]*self.z_am[k,w] for w in self.N_am) + y >= quicksum(self.base_r_demand_am[u]*self.x[u,k] for u in self.V) for k in self.K)
        mdl.addConstrs(quicksum(self.z_am[k,w] for k in self.K)==1 for w in self.N_am)
        mdl.addConstrs(quicksum(self.z_am[k,w] for w in self.N_am)>=L_min_am for k in self.K)
        mdl.addConstrs(quicksum(self.z_am[k,w] for w in self.N_am)<=L_max_am for k in self.K)

        # for pm nurses
        mdl.addConstrs( quicksum(self.base_hcp_load[w]*self.z_pm[k,w] for w in self.N_pm) + y >= quicksum(self.base_r_demand_pm[u]*self.x[u,k] for u in self.V) for k in self.K)
        mdl.addConstrs(quicksum(self.z_pm[k,w] for k in self.K)==1 for w in self.N_pm)
        mdl.addConstrs(quicksum(self.z_pm[k,w] for w in self.N_pm)>=L_min_pm for k in self.K)
        mdl.addConstrs(quicksum(self.z_pm[k,w] for w in self.N_pm)<=L_max_pm for k in self.K)
        mdl.addLConstr(y<=self.args.Y)

        # Save the linear program 
        Path("output/LP/").mkdir(parents=True, exist_ok=True)
        
        mdl.optimize()
        print("Objective value of ILP is :", mdl.objVal)
        
        if self.is_feasible_solution():
            self.retrieve_linear_program_results()
            return True
        else:
            return False


    def is_feasible_solution(self):
        feasible = True
        for u in self.V:
            for v in self.V:
                if (u,v) in self.e and int(self.e[u,v].x)==0: # same bubble
                    same_bubble = False
                    for k in self.K:
                        if int(self.x[u,k].x)==1 and int(self.x[v,k].x)==1:
                            same_bubble = True
                            break
                    feasible = same_bubble
                elif (u,v) in self.e and int(self.e[u,v].x)==1: # different bubble
                    for k in self.K:
                        if int(self.x[u,k].x)==1 and int(self.x[v,k].x)==1:
                            feasible = False
                            break            
        return feasible



    def retrieve_linear_program_results(self):
        # Room partition
        self.clustered_rooms = {}
        for k in self.K:
            for v in self.V:
                if self.x[v,k].x>0:
                    if k-1 not in self.clustered_rooms:
                        self.clustered_rooms[k-1] = []
                    self.clustered_rooms[k-1].append(v)

        # AM hcp partition
        self.clustered_am_nrs = {}
        for k in self.K:
            for w in self.N_am:
                if k not in self.clustered_am_nrs:
                    self.clustered_am_nrs[k] = []
                if self.z_am[k,w].x>0:
                    self.clustered_am_nrs[k].append(w)
        # PM hcp partition
        self.clustered_pm_nrs = {}
        for k in self.K:
            for w in self.N_pm:
                if k not in self.clustered_pm_nrs:
                    self.clustered_pm_nrs[k] = []
                if self.z_pm[k,w].x>0:
                    self.clustered_pm_nrs[k].append(w)
        self.clustered_am_nrs = {key-1:val for key,val in self.clustered_am_nrs.items()}
        self.clustered_pm_nrs ={key-1:val for key,val in self.clustered_pm_nrs.items()}
        

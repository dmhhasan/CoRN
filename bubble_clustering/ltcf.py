from .base import *

class LTCFBubbleClustering(BaseClustering):
    def __init__(self, args, h_list, r_list, h_type, r_type, r_dist, h_visits):
        #super().__init__(self, args, r_list, h_list)
        self.args = args 
        self.h_list = h_list  # HCP list
        #self.r_list = r_list  # Room list
        self.h_type = h_type  # HCP type e.g. am_nurse, pm_nurse, non_nurse
        self.r_type = r_type  # Room type
        self.r_dist = r_dist  # Room to room distances in hops (1 hop ~ 3 meter)
        self.h_visits = h_visits
        self.nrs_list = [h for h in self.h_list if self.h_type[h]=='nurse']
        self.n_bubble = args.n_bubble
        self.bbl_list = np.arange(self.n_bubble)
        
        # In LTCF data there are multiple types of rooms;
        # For Bubble Clustering only pick rooms based on HCP mobility to resident rooms
        self.r_list = set()
        for i in range(len(h_visits)):
            hcp, room ,start, end = h_visits[i]
            self.r_list.add(room)
        self.r_list = list(self.r_list)


    def find_baseline_load_demand_mobility(self):
        self.base_r_demand= {}
        self.base_hcp_load = {}
        self.base_h_mobility = {}
        self.base_visit_graph = {} # Create a mobility graph for 30 days

        # Initialize all HCP's load, mobility and patient room demands
        for h in self.h_list:
            self.base_hcp_load[h] = 0
            self.base_h_mobility[h] = 0
        for r in self.r_list:
            self.base_r_demand[r] = 0

        last_visit = {}
        for h in self.h_list:
            last_visit[h] = {'start':0, 'end':0, 'room': None}

        # Repeat 1 day HCP-Room interactions over 30 days patients stays
        #for day in range(1,31):
        for i in range(len(self.h_visits)):
            cur_hcp, cur_room, h_start, h_end = self.h_visits[i]
            # Update visit graph (only for non-substituble HCPs i.e. HCP who do not belong to any bubble)
            if self.h_type[cur_hcp] == 'non-nurse':
                key = (cur_hcp, cur_room)
                if key not in self.base_visit_graph:
                    self.base_visit_graph[key] = []
                self.base_visit_graph[key].append([h_start, h_end])
            # Update load, demand and mobility
            if self.h_type[cur_hcp]=='nurse':
                # update load and met demand here
                val = (h_end - h_start)/60.0
                self.base_hcp_load[cur_hcp]+= val
                self.base_r_demand[cur_room]+=val
                self.base_h_mobility[cur_hcp] += self.r_dist[(last_visit[cur_hcp]['room'], cur_room)] if last_visit[cur_hcp]['room'] is not None else 0
                
                # Keep track of the last visit
                last_visit[cur_hcp] = {'start':h_start, 'end':h_end, 'room':cur_room}
        return 0 


    # Overriding abstract method 
    def create_random_bubbles(self):
        # Use seed to generate different clustering for different experiments
        random.seed(datetime.now())

        # Random permutation
        random.shuffle(self.nrs_list)
        random.shuffle(self.r_list)
        random.shuffle(self.bbl_list)
        
        nrs_idx = 0
        r_idx = 0
        bbl_idx = 0
        
        # Hold information of which hcp belongs to which bubble
        self.h_bubble = {}
        self.h_per_room = {}
        self.clustering = {}
        for b in self.bbl_list:
            self.clustering[b]={'nurse':[], 'room':[]}

        while True:
            b = self.bbl_list[bbl_idx]
            if nrs_idx < len(self.nrs_list):
                self.clustering[b]['nurse'].append(self.nrs_list[nrs_idx])
                self.h_bubble[self.nrs_list[nrs_idx]] = b
                nrs_idx+=1
            if r_idx < len(self.r_list):
                self.clustering[b]['room'].append(self.r_list[r_idx])
                r_idx+=1
            bbl_idx +=1
            if bbl_idx==len(self.bbl_list):
                bbl_idx=0
            if nrs_idx==len(self.nrs_list) and r_idx==len(self.r_list):
                break
        self.clustered_rooms = {}
        for key in self.clustering:
            self.clustered_rooms[key] = self.clustering[key]['room']

        for key in self.clustering.keys():
            rooms = self.clustering[key]['room']
            nrs = self.clustering[key]['nurse']
            for r in rooms:
                if r not in self.h_per_room.keys():
                    self.h_per_room[r] = {"nurse":[]}
                for n in nrs:
                    self.h_per_room[r]["nurse"].append(n)

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
                self.clustering[b]={'nurse':[], 'room':[]}

            for b in self.bbl_list:
                for n in self.clustered_nurses[b]:
                    self.h_bubble[n] = b

            for b in self.bbl_list:
                self.clustering[b]['nurse'] = self.clustered_nurses[b]
            
            for b in self.bbl_list:
                self.clustering[b]['room'] = self.clustered_rooms[b]

            for key in self.clustering.keys():
                rooms = self.clustering[key]['room']
                nurses = self.clustering[key]['nurse']
                for r in rooms:
                    if r not in self.h_per_room.keys():
                        self.h_per_room[r] = {"nurse":[]}
                    for n in nurses:
                        self.h_per_room[r]["nurse"].append(n)
            self.store_bubble_clustering()

        return self.h_bubble, self.h_per_room, self.clustering



    def solve_integer_linear_program(self):
        # Solve ILP problem for graph partitioning
        # Find n_bubble partitions such that edge-cut is minimized in terms of edge weights
        
        # First create edge variables for all edges in graph
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

        L_max_nurse = math.ceil(len(self.nrs_list)/self.n_bubble)
        L_min_nurse = math.floor(len(self.nrs_list)/self.n_bubble)

        self.N_all = self.nrs_list

        mdl = Model('CoRN')
        self.e= mdl.addVars(E, vtype=GRB.BINARY)
        self.x= mdl.addVars(self.V,self.K, vtype=GRB.BINARY)
        d= mdl.addVar(vtype=GRB.CONTINUOUS)
        # for nurses
        self.z = mdl.addVars(self.K,self.N_all, vtype=GRB.BINARY)
        y = mdl.addVar(vtype=GRB.CONTINUOUS)
        mdl.modelSense = GRB.MINIMIZE

        mdl.setObjective(quicksum(self.e[u,v]*W[u,v] for u,v in E)) # Objective

        mdl.addConstrs(self.e[u,v]>=(self.x[u,k]-self.x[v,k]) for k in self.K for u,v in E)
        mdl.addConstrs(self.e[u,v]>=(self.x[v,k]-self.x[u,k]) for k in self.K for u,v in E)

        mdl.addConstrs(self.r_dist[(u,v)]*(1-self.e[u,v])<=d for u,v in E)

        mdl.addConstrs(quicksum(self.x[v,k]*C[v] for v in self.V)<=L_max for k in self.K)
        mdl.addConstrs(quicksum(self.x[v,k]*C[v] for v in self.V)>=L_min for k in self.K)
        mdl.addConstrs(quicksum(self.x[v,k] for k in self.K)==1 for v in self.V)
        mdl.addConstr(d<=self.args.D)

        mdl.addConstrs( quicksum(self.base_hcp_load[w]*self.z[k,w] for w in self.N_all) + y >= quicksum(self.base_r_demand[u]*self.x[u,k] for u in self.V) for k in self.K)
        mdl.addConstrs(quicksum(self.z[k,w] for k in self.K)==1 for w in self.N_all)
        mdl.addConstrs(quicksum(self.z[k,w] for w in self.N_all)>=L_min_nurse for k in self.K)
        mdl.addConstrs(quicksum(self.z[k,w] for w in self.N_all)<=L_max_nurse for k in self.K)
        mdl.addConstr(y<=self.args.Y)

        # Save the linear program 
        Path("output/LP/").mkdir(parents=True, exist_ok=True)
        mdl.write("output/LP/CoRN_"+self.args.data+"_Linear_Program.lp")
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
        self.clustered_rooms = {}
        # Room clustering
        for k in self.K:
            for v in self.V:
                if self.x[v,k].x>0:
                    if k-1 not in self.clustered_rooms:
                        self.clustered_rooms[k-1] = []
                    self.clustered_rooms[k-1].append(v)

        # Nurse clustering
        self.clustered_nurses = {}
        for k in self.K:
            for w in self.N_all:
                if k not in self.clustered_nurses:
                    self.clustered_nurses[k] = []
                if self.z[k,w].x>0:
                    self.clustered_nurses[k].append(w)

        self.clustered_nurses = {key-1:val for key,val in self.clustered_nurses.items()}
        

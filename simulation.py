import random
import multiprocessing as mp
import numpy as np
from datetime import datetime, timedelta
import math
import pathlib
from utils import dump_load_pickle_object

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

class LoadData:
    def __init__(self, load, trans_prob):
        self.load = load
        self.trans_prob = trans_prob

class MobilityData:
    def __init__(self, mobility, trans_prob):
        self.mobility = mobility
        self.trans_prob = trans_prob

class DemandData:
    def __init__(self, met_demand_am, met_demand_pm, unmet_demand_am, unmet_demand_pm, trans_prob):
        self.met_demand_am = met_demand_am
        self.met_demand_pm = met_demand_pm
        self.unmet_demand_am = unmet_demand_am
        self.unmet_demand_pm = unmet_demand_pm
        self.trans_prob = trans_prob


def init_load(hcp_list, load_list, trans_prob):
    for h in hcp_list:
        update_load(load_list=load_list, hcp_id=h, load=0, trans_prob=trans_prob)

def init_mobility(hcp_list, mobility_list, trans_prob):
    for h in hcp_list:
        update_mobility(mobility_list=mobility_list, hcp_id=h, mobility=0, trans_prob=trans_prob)
                
def init_demand(room_list, demand_list, trans_prob):
    for r in room_list:
        update_demand(demand_list=demand_list, room_id=r, met_demand_am=0, met_demand_pm=0, unmet_demand_am=0, unmet_demand_pm=0, trans_prob=trans_prob)
               
def update_demand(demand_list, room_id, met_demand_am, met_demand_pm, unmet_demand_am, unmet_demand_pm, trans_prob):
    if room_id not in demand_list:
        demand_list[room_id] = DemandData(met_demand_am=met_demand_am, met_demand_pm = met_demand_pm, unmet_demand_am=unmet_demand_am, unmet_demand_pm=unmet_demand_pm, trans_prob=trans_prob)
    else:
        demand_list[room_id].met_demand_am+=met_demand_am
        demand_list[room_id].met_demand_pm+=met_demand_pm
        demand_list[room_id].unmet_demand_am+=unmet_demand_am
        demand_list[room_id].unmet_demand_pm+=unmet_demand_pm

def update_load(load_list, hcp_id, load, trans_prob):
    if hcp_id not in load_list:
        load_list[hcp_id] = LoadData(load=load, trans_prob=trans_prob)
    else:
        load_list[hcp_id].load+=load

def update_mobility(mobility_list, hcp_id, mobility, trans_prob):
    if hcp_id not in mobility_list:
        mobility_list[hcp_id] = MobilityData(mobility=mobility, trans_prob=trans_prob)
    else:
        mobility_list[hcp_id].mobility+=mobility

def add_edge(graph, edge, node_type, edge_type, edge_label, contact_loc, ts_start, ts_end):
    nodeA, nodeB = edge[0], edge[1]
    if nodeA not in graph:
        graph.add_node(nodeA, node_type=node_type[0])
    if nodeB not in graph:
        graph.add_node(nodeB, node_type=node_type[1])
    graph.add_edge(nodeA, nodeB, edge_type = edge_type, edge_label = edge_label, contact_loc = contact_loc, ts_start = ts_start, ts_end = ts_end)

def init_last_visits(simulationData):
    # Last visit keeps track of most recent visit of every HCP; 
    # initially last visit (start, end) time is set to the earliest 
    # time found in the real data; time must be normalized
    last_visit = {}
    first_visit = simulationData.hcp_visits[0]
    beginning_time = first_visit[2] # since visits are sorted by start time
    normalized_beginning_time = get_normalized_time(beginning_time, simulationData.interaction_date,1)
    for h in simulationData.hcp_list:
        last_visit[h] = {'start':normalized_beginning_time, 'end':normalized_beginning_time, 'room': None}
    return last_visit

def get_normalized_time(time, start_date_main, day):
    # As a certain day's events are replicated over 30 days, for each day the start and end times of these events have to be
    # normalized. This is done by subtracting (current shift date - 1) times 1440 minutes from the start and end times and then,
    # adding a multiplier times 1440 value to the normalized dates for each day.
    return (time - timedelta(minutes=(start_date_main.day - 1) * 1440) + timedelta(
        minutes=day * 1440))

def get_patient(patients_stay, cur_room, normalised_cur_start_time, normalised_cur_end_time):
    patient_id = None
    for i in range(patients_stay.shape[0]):
        patient, room, start_t, end_t = patients_stay[i]
        if room == cur_room and start_t <= normalised_cur_start_time <= end_t and start_t <= normalised_cur_end_time <= end_t:
            patient_id = patient
            break
    return patient_id

def find_overlapped_contact(cur_room, cur_start_time, cur_end_time, hcp_visits):
    overlapped_contact_index = []
    for i in range(hcp_visits.shape[0]):
        hcp, room, start_t, end_t = hcp_visits[i]
        if (cur_room == room and ( cur_start_time<=start_t<=cur_end_time) ): # dont add  cur_start_time<=e<=cur_end_time to avoid duplicate
            overlapped_contact_index.append(i)
    return overlapped_contact_index

def get_random_nurse(cur_room, hcws_per_room, hcp_type, hcp_availability, normalised_cur_start_time):
    filter_list = []
    for h in  hcws_per_room[cur_room][hcp_type]:
        if hcp_availability[h]<normalised_cur_start_time:
            filter_list.append(h)
    return filter_list[random.randint(0,len(filter_list)-1)] if len(filter_list)>0 else None

def find_overlapping_period(startA, endA, startB, endB):
    latest_start = max(startA, startB)
    earliest_end = min(endA, endB)
    delta = (earliest_end - latest_start).total_seconds()
    overlap = max(0, delta)
    return overlap

def get_env_contamination(cur_hcp, hcp_type, cur_time, last_visit, infection_list, hcp_bubble):
    cur_shift_hcp_list = []
    if 'am' in hcp_type[cur_hcp]:
        cur_shift_hcp_list = [h for h in hcp_type if hcp_type[h]=='am_nurse']
    elif 'pm' in hcp_type[cur_hcp]:
        cur_shift_hcp_list = [h for h in hcp_type if hcp_type[h]=='pm_nurse']

    outside_start = last_visit[cur_hcp]['end'] # last_visit should not be the current visit of cur_hcp 
    outside_end = cur_time

    # Calculate overlapping between cur_hcp and
    # other cur shift HCP's during the outside room time period
    
    not_mixing_trans_prob = 1
    for h in cur_shift_hcp_list:
        # if the outside hcp not contagious 
        # chance of transmission is zero
        if h==cur_hcp or h not in infection_list:
            continue
        start = last_visit[h]['start']
        end = last_visit[h]['end']

        if (outside_end-outside_start).total_seconds()>60*60:
            outside_start = outside_end

        #overlapping = max(0, (start-outside_start).total_seconds()) + max(0, min((outside_end-end).total_seconds(), (outside_end-outside_start).total_seconds()))
        overlapping = find_overlapping_period(startA=start, endA=end, startB=outside_start, endB=outside_end)
        if overlapping!=0:
            prob = overlapping/((outside_end-outside_start).total_seconds())
        else:
            prob = 0
        if hcp_bubble is not None and h in hcp_bubble and cur_hcp in hcp_bubble and hcp_bubble[h] == hcp_bubble[cur_hcp]:
            mixing_trans_prob = 0.0005*prob
        else:
            mixing_trans_prob = 0.0005 * 0.75 *prob # rho = 0.75, intra bubble mixing trans prob = 0.0005
        long_mixing_trans_prob = mixing_trans_prob#1 - math.pow((1 - mixing_trans_prob), math.ceil(overlapping / 30)) # 30 seconds

        not_mixing_trans_prob *=(1-long_mixing_trans_prob)
    env_contamination = 1-not_mixing_trans_prob
    return env_contamination

def get_long_exposure_prob(cur_start_time, cur_end_time, trans_prob):
    visit_duration = (cur_end_time - cur_start_time).total_seconds()
    long_exposure_trans_prob = 1 - math.pow((1 - trans_prob), math.ceil(visit_duration / 30)) # 30 seconds
    return long_exposure_trans_prob

def store_rewired_network(args, adj, n_bubble, node_list, simulation): 
    isolated_list = list(np.where(~adj.any(axis=1))[0])
    pathlib.Path("output/rewired_network/"+args.data+"/"+str(simulation)+"/").mkdir(parents=True, exist_ok=True)
    node_arr = np.array(node_list)
    np.savez("output/rewired_network/"+args.data+"/"+str(simulation)+"/"+"bubble_"+str(n_bubble)+"_adjacency.npz", name1=adj, name2=node_arr)

def store_baseline_load_demand_mobility(args, load_list, demand_list, mobility_list):
    load_dic = {}
    if args.data == 'MICU':
        demand_am_dic = {}
        demand_pm_dic = {}
    else:
        demand_dic = {}
    mobility_dic = {}

    for hcp in load_list.keys():
        load_dic[hcp] = load_list[hcp].load
    for room in demand_list.keys():
        if args.data == 'MICU':
            demand_am_dic[room] = demand_list[room].met_demand_am
            demand_pm_dic[room] = demand_list[room].met_demand_pm
        else:
            demand_dic[room] = demand_list[room].met_demand
    for hcp in mobility_list.keys():
        mobility_dic[hcp] = mobility_list[hcp].mobility

    pathlib.Path("dumps/load/"+args.data+"/").mkdir(parents=True, exist_ok=True)
    pathlib.Path("dumps/demand/"+args.data+"/").mkdir(parents=True, exist_ok=True)
    pathlib.Path("dumps/mobility/"+args.data+"/").mkdir(parents=True, exist_ok=True)
    dump_load_pickle_object(action='dump', filename= "dumps/load/"+args.data+"/baseline_hcp_load", data= load_dic)
    if args.data == 'MICU':
        dump_load_pickle_object(action='dump', filename= "dumps/demand/"+args.data+"/baseline_room_am_demand", data= demand_am_dic)
        dump_load_pickle_object(action='dump', filename= "dumps/demand/"+args.data+"/baseline_room_pm_demand", data= demand_pm_dic)
    else:
        dump_load_pickle_object(action='dump', filename= "dumps/demand/"+args.data+"/baseline_room_pm_demand", data= demand_dic)
    dump_load_pickle_object(action='dump', filename= "dumps/mobility/"+args.data+"/baseline_hcp_mobility", data= mobility_dic)

def run_MICU_simulation(args, hcp_visits, hcp_list, room_list, patients_stay, hcp_type, room_distances, clustering_data):
    interaction_date = datetime(2011, 6, 3)
    am_nurse_list = [hcp for hcp in hcp_list if hcp_type[hcp]=='am_nurse']
    pm_nurse_list = [hcp for hcp in hcp_list if hcp_type[hcp]=='pm_nurse']
    non_nurse_list = [hcp for hcp in hcp_list if 'nurse' not in hcp_type[hcp]]
    
    trans_probs = args.trans_probs
    
    data = []

    # Corresponding  replicates in different assignments need to have
    # the same infection source 
    random.seed(12)
    if args.inf_source == "nurse":
        inf_srcs = random.choices(am_nurse_list+pm_nurse_list, k=args.n_replicate)
    elif args.inf_source == 'non-nurse':
        inf_srcs = random.choices(non_nurse_list, k=args.n_replicate)

    for j in range(args.n_replicate):
        for i in range(len(trans_probs)):
            transmissibility = trans_probs[i]
            if args.simulation == 'rewired_ilp':
                hcp_bubble = clustering_data[0][0]
                hcws_per_room = clustering_data[0][1]
                b_assignment = clustering_data[0][2]
            elif args.simulation == 'rewired_random':
                hcp_bubble = clustering_data[j][0]
                hcws_per_room = clustering_data[j][1]
                b_assignment = clustering_data[0][2]
            elif args.simulation == 'base':
                hcp_bubble = None
                hcws_per_room = None
                b_assignment = None

            simulation_data = SimulationData(args = args,
                                            trans_prob = transmissibility, 
                                            hcp_visits = hcp_visits, 
                                            patients_stay = patients_stay,
                                            hcp_type = hcp_type,
                                            hcp_list = hcp_list, 
                                            room_list = room_list.tolist(), 
                                            am_nurse_list = am_nurse_list, 
                                            pm_nurse_list = pm_nurse_list, 
                                            non_nurse_list = non_nurse_list,
                                            hcp_bubble= hcp_bubble,
                                            hcws_per_room = hcws_per_room,
                                            b_assignment = b_assignment,
                                            inf_src = inf_srcs[j],
                                            replicate_id = j,
                                            room_distances = room_distances,
                                            interaction_date = interaction_date
                                           )
            data.append(simulation_data)
    pool = mp.Pool(args.n_cpu)

    results = pool.map(covid19_simulation, data)
    pool.close()
    pool.join()
    return results

#  dump = open("dumps/load/"+self.args.data+"/baseline_hcp_load", "rb")
#         self.base_hcp_load = pickle.load(dump)
#         dump.close()

#         dump = open("dumps/demand/"+self.args.data+"/baseline_room_am_demand", "rb")
#         self.base_r_demand_am = pickle.load(dump)
#         dump.close()

#         dump = open("dumps/demand/"+self.args.data+"/baseline_room_pm_demand", "rb")
#         self.base_r_demand_pm = pickle.load(dump)
#         dump.close()


# This is a SEIR based COVID19 model
def covid19_simulation(sdata):
    args = sdata.args
    random.seed(datetime.now())

    #Save an adjacency matrix of post-simulation i.e. rewired contact network
    if args.simulation == 'rewired_ilp' or args.simulation == 'rewired_random':
        node_list = sdata.non_nurse_list+sdata.am_nurse_list+sdata.pm_nurse_list+sdata.room_list
        node_mapping = {}
        reverse_node_mapping = {}
        for item in node_list:
            reverse_node_mapping[len(node_mapping)] = item
            node_mapping[item] = len(node_mapping)
        adj = np.zeros((len(node_list),len(node_list)))

    #  W is the "wait time" or incubation period, in
    # days, and T is the temporal duration, also in days. An infection
    # occurs (early) on day 0; symptoms emerge on day W, and then the
    # infection lasts through day W+T.    
    W=5
    T=10

    # The first, the exp/exp model, shedding ramps up exponentially
    # (faster) from day 0 and peaks at day W, then exponentially ramps down
    # (slower) for T days through day W+T-1. Symptoms emerge at peak shedding.
    MODEL_TYPE_UNI_UNI = 'uni_uni'
    MODEL_TYPE_EXP_EXP = 'exp_exp'
    shedding_dic = {}
    shedding_dic['exp_exp'] = [0.001, 0.0039, 0.0156, 0.0625, 0.25, 1, 0.5, 0.25, 0.125, 0.0625, 0.0312, 0.0156, 0.0078, 0.0039, 0.001,0]
    shedding_dic['uni_uni'] = [0.01, 0.01, 0.01, 0.01, 0.01, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    shedding_scale = .1
    model_type = MODEL_TYPE_EXP_EXP

    STATUS_INFECTED = 'infected'
    STATUS_SHEDDING = 'shedding'
    STATUS_QUARANTINE = 'qurantine'
    STATUS_RECOVERED = 'recovered'

    # Record different cost values
    infection_list = {}
    demand_list = {}
    load_list = {}
    mobility_list = {}

    # Initialize all HCP's load and patient room demands
    # Initial demands/loads are zero
    init_load(hcp_list=sdata.hcp_list,load_list=load_list, trans_prob=sdata.trans_prob)
    init_demand(room_list=sdata.room_list, demand_list=demand_list, trans_prob=sdata.trans_prob)
    init_mobility(hcp_list=sdata.hcp_list, mobility_list=mobility_list, trans_prob=sdata.trans_prob)

    rand_hcp = sdata.inf_src
    infectionData = InfectionData(inf_day=0,is_first_infected = True, infected_by=None, trans_prob=sdata.trans_prob, infected_at=None, status=STATUS_SHEDDING)
    infection_list[rand_hcp] = infectionData
 
    # For rewired simulation,  use bubble clustering
    if args.simulation == 'rewired_ilp' or args.simulation=='rewired_random':
        src_bubble = sdata.hcp_bubble[rand_hcp] # infection source is nurse and nurse present in hcp_bubble
        ext_leave_cnt = 0
        ext_reach_cnt = 0
        hcp_availability = {}
        for item in sdata.hcp_list:
            hcp_availability[item] = datetime(1970, 1, 1)

    last_visit = init_last_visits(simulationData=sdata)
    
    if args.simulation == 'base':
        visit_history = {}


    for day in range(1,31):
        random.seed(datetime.now())

        # Update status 
        for key in infection_list.keys():
            if model_type == MODEL_TYPE_EXP_EXP:
                if infection_list[key].inf_day==1:
                    infection_list[key].status = STATUS_SHEDDING
                elif infection_list[key].inf_day == W+T:
                    infection_list[key].status = STATUS_RECOVERED

        # Keep track of HCP_Patient interaction index that are processed together 
        # for overlapping visits in the same room at the same time
        overlapped_processed_contact = []

        for i in range(len(sdata.hcp_visits)):
            if i in overlapped_processed_contact:
                continue
            cur_hcp, cur_room, cur_start_time, cur_end_time = sdata.hcp_visits[i]

            normalised_cur_start_time = get_normalized_time(cur_start_time, sdata.interaction_date, day)
            normalised_cur_end_time = get_normalized_time(cur_end_time, sdata.interaction_date, day)

            # Determine if HCW-patient contact corresponds to any real patient contact
            patient_id = get_patient(sdata.patients_stay, cur_room, normalised_cur_start_time, normalised_cur_end_time)


            if patient_id is not None:
                overlapped_contact_index = find_overlapped_contact(cur_room=cur_room, cur_start_time=cur_start_time,
                                                         cur_end_time=cur_end_time, hcp_visits= sdata.hcp_visits)
                
                # In perturb simulation, HCP of an index can be randomly replaced
                # Therefore, keep a mapping which index maps to which HCP
                overlapped_index_to_hcp_mapping = {}
                
                for index in overlapped_contact_index:
                    overlapped_processed_contact.append(index)
                    index_hcp = sdata.hcp_visits[index][0] #0->hcp id
                    index_start = sdata.hcp_visits[index][2] #2->start_time
                    index_end = sdata.hcp_visits[index][3] #3->end_time
                    index_normalized_end = get_normalized_time(index_end, sdata.interaction_date,day)
                    index_normalized_start = get_normalized_time(index_start, sdata.interaction_date,day)
                                
                    if args.simulation != "base":
                        # perturb only nurses
                        if 'nurse' in sdata.hcp_type[index_hcp]:
                            random_h = get_random_nurse(cur_room = cur_room, hcws_per_room= sdata.hcws_per_room, 
                                            hcp_type = sdata.hcp_type[index_hcp], hcp_availability= hcp_availability, 
                                            normalised_cur_start_time = index_normalized_start)

                            if random_h is not None:
                                overlapped_index_to_hcp_mapping[index] = random_h
                                hcp_availability[random_h] = index_normalized_end
                            else:
                                overlapped_index_to_hcp_mapping[index] = None
                                unmet_demand = (index_normalized_end - index_normalized_start).total_seconds()/60.0 
                                # If the original HCP is a am_nurse or am_doctor
                                # the unmet demand would be unmet_demand_am in that case
                                if 'am' in sdata.hcp_type[index_hcp]:
                                    update_demand(demand_list=demand_list, room_id=cur_room, met_demand_am=0, met_demand_pm=0, unmet_demand_am=unmet_demand, unmet_demand_pm=0, trans_prob=sdata.trans_prob)
                                elif 'pm' in sdata.hcp_type[index_hcp]:
                                    update_demand(demand_list=demand_list, room_id=cur_room, met_demand_am=0, met_demand_pm=0, unmet_demand_am=0, unmet_demand_pm=unmet_demand, trans_prob=sdata.trans_prob)
                                # For ccstaff, offunit there is no am or pm in HCP type
                                # therefore in such case determine the time by hour
                                elif index_start.hour<12:
                                    update_demand(demand_list=demand_list, room_id=cur_room, met_demand_am=0, met_demand_pm=0, unmet_demand_am=unmet_demand, unmet_demand_pm=0, trans_prob=sdata.trans_prob)
                                elif index_start.hour>=12:
                                    update_demand(demand_list=demand_list, room_id=cur_room, met_demand_am=0, met_demand_pm=0, unmet_demand_am=0, unmet_demand_pm=unmet_demand, trans_prob=sdata.trans_prob)
                                
                        else:
                            overlapped_index_to_hcp_mapping[index] = index_hcp
                    else:
                        overlapped_index_to_hcp_mapping[index] = index_hcp
                    # mapped_hcp could be either same or different from index_hcp
                    # depending on if perturb simulation and nurse type
                    mapped_hcp = overlapped_index_to_hcp_mapping[index]
                    # mapped_hcp is None only when demand is unmet.
                    # don't need update unmet demand here as it is already done above
                    if mapped_hcp is None:
                        continue

                    # keep visit history of natural simulation to calculate probabilistic edge weights 
                    # between rooms in bubble partitioning
                    if args.simulation == 'base':
                        if (mapped_hcp, cur_room) not in visit_history:
                            visit_history[(mapped_hcp, cur_room)] = []
                        visit_history[(mapped_hcp, cur_room)].append([index_normalized_start, index_normalized_end])

                    # #Save contact into adjacency matrix
                    if args.simulation != 'base':
                        adj[node_mapping[mapped_hcp],node_mapping[cur_room]]+= (index_end - index_start).total_seconds()
                        adj[node_mapping[cur_room],node_mapping[mapped_hcp]]+= (index_end - index_start).total_seconds()

                    # update load and met demand here
                    load = (index_end - index_start).total_seconds()/60.0
                    update_load(load_list=load_list, hcp_id=mapped_hcp, load=load, trans_prob=sdata.trans_prob)
                    met_demand = (index_end - index_start).total_seconds()/60.0
                    if 'nurse' in sdata.hcp_type[mapped_hcp]:
                        if 'am' in sdata.hcp_type[mapped_hcp]:
                            update_demand(demand_list=demand_list, room_id=cur_room, met_demand_am=met_demand, met_demand_pm=0, unmet_demand_am=0,unmet_demand_pm=0, trans_prob=sdata.trans_prob)
                        elif 'pm' in sdata.hcp_type[mapped_hcp]:
                            update_demand(demand_list=demand_list, room_id=cur_room, met_demand_am=0, met_demand_pm=met_demand, unmet_demand_am=0,unmet_demand_pm=0, trans_prob=sdata.trans_prob)
                        elif index_start.hour<12:
                            update_demand(demand_list=demand_list, room_id=cur_room, met_demand_am=met_demand, met_demand_pm=0, unmet_demand_am=0,unmet_demand_pm=0, trans_prob=sdata.trans_prob)
                        elif index_start.hour>=12:
                            update_demand(demand_list=demand_list, room_id=cur_room, met_demand_am=0, met_demand_pm=met_demand, unmet_demand_am=0,unmet_demand_pm=0, trans_prob=sdata.trans_prob)
                        
                    mobility = sdata.room_distances[(last_visit[mapped_hcp]['room'], cur_room)] if last_visit[mapped_hcp]['room'] is not None else 0
                    update_mobility(mobility_list=mobility_list, hcp_id=mapped_hcp, mobility=mobility, trans_prob=sdata.trans_prob)
                    
                    # calculate long_exposure_trans_prob for mapped_hcp
                    long_exposure_trans_prob = get_long_exposure_prob(index_start, index_end, sdata.trans_prob)
                    # calculate environmental contamination if not infected
                    if mapped_hcp not in infection_list:
                        if args.simulation != 'base':
                            env_contamination = get_env_contamination(cur_hcp=mapped_hcp, hcp_type=sdata.hcp_type,
                                                                cur_time=index_normalized_start, 
                                                                last_visit=last_visit, infection_list=infection_list,
                                                                hcp_bubble=sdata.hcp_bubble)
                        else:
                            env_contamination = get_env_contamination(cur_hcp=mapped_hcp, hcp_type=sdata.hcp_type,
                                                                cur_time=index_normalized_start, 
                                                                last_visit=last_visit, infection_list=infection_list,
                                                                hcp_bubble=None)
                        if random.random() < env_contamination:
                            # HCP get infected from environmental exposure
                            infectionData = InfectionData(inf_day=0,is_first_infected = False, infected_by="env", trans_prob=sdata.trans_prob, infected_at="env", status=STATUS_INFECTED)
                            infection_list[mapped_hcp] = infectionData
                            # if args.simulation != 'baseline':
                            #     if mapped_hcp in sdata.hcp_bubble and sdata.hcp_bubble[mapped_hcp]!=src_bubble:
                            #         ext_leave_cnt+=1
                            #     elif mapped_hcp not in sdata.hcp_bubble:
                            #         ext_leave_cnt+=1

                        elif patient_id in infection_list.keys() and infection_list[patient_id].status == STATUS_SHEDDING:#and random.random() < long_exposure_trans_prob:
                            shedding = shedding_dic[model_type][infection_list[patient_id].inf_day]/shedding_scale
                            shedding*=sdata.trans_prob
                            scaled_shedding = get_long_exposure_prob(index_start, index_end, shedding)
                            if random.random() < scaled_shedding:
                                # HCP get infected by a patient
                                infectionData = InfectionData(inf_day=0,is_first_infected = False, infected_by="patient", trans_prob=sdata.trans_prob, infected_at=cur_room, status=STATUS_INFECTED)
                                infection_list[mapped_hcp] = infectionData
                                infection_list[patient_id].secondary_infs.append(mapped_hcp)
                                if args.simulation != 'base':
                                    if cur_room not in sdata.b_assignment[src_bubble]:
                                        print("one")
                                        ext_leave_cnt+=1
                                        ext_reach_cnt+=1
                                    else:
                                        if mapped_hcp not in sdata.hcp_bubble: # non-nurse
                                            ext_leave_cnt+=1
                                        elif sdata.hcp_bublle[mapped_hcp] != src_bubble:
                                            ext_leave_cnt+=1
                                # Update edge type from 'contact' to 'transmission'
                                #update_edge_type(graph=graph, edge=[patient_id, mapped_hcp], node_type = ['patient', sdata.hcp_type[mapped_hcp]],
                                #                     edge_type = "transmission", edge_label= 'patient->hcp', contact_loc = cur_room, ts_start = index_normalized_start, ts_end = index_normalized_end)
                # find the probability of patient being infected
                # by overlapped HCPs in the room 
                if patient_id not in infection_list.keys():
                    patient_trans_prob = 1
                    infected_by = list() # which HCPs infect the patient and for how long it contacts
                    for index in overlapped_contact_index:
                        index_start = sdata.hcp_visits[index][2] #2->start_time
                        index_end = sdata.hcp_visits[index][3] #3->end_time
                        index_normalized_end = get_normalized_time(index_end, sdata.interaction_date,day)
                        index_normalized_start = get_normalized_time(index_start, sdata.interaction_date,day)
                    
                        mapped_hcp = overlapped_index_to_hcp_mapping[index]
                        if mapped_hcp == None:
                            continue
                        if mapped_hcp in infection_list and infection_list[mapped_hcp].status == STATUS_SHEDDING:
                            shedding = shedding_dic[model_type][infection_list[mapped_hcp].inf_day]/shedding_scale
                            shedding*=sdata.trans_prob
                            scaled_shedding = get_long_exposure_prob(index_start, index_end, shedding)
                            patient_trans_prob*=(1 - scaled_shedding)
                            infected_by.append([mapped_hcp, [index_normalized_start, index_normalized_end] ])

                    patient_trans_prob = (1-patient_trans_prob)
                    if random.random() < patient_trans_prob:
                        # Patient get infected by HCP
                        infectionData = InfectionData(inf_day=0,is_first_infected = False, infected_by="hcp", trans_prob=sdata.trans_prob, infected_at = cur_room, status=STATUS_INFECTED)
                        infection_list[patient_id] = infectionData
                        if args.simulation != 'base':
                            if cur_room not in sdata.b_assignment[src_bubble]:
                                    print("two")
                                    ext_leave_cnt+=1
                                    ext_reach_cnt+=1
                        for h,contact_timestamp in infected_by:
                            infection_list[h].secondary_infs.append(patient_id)
                            #update_edge_type(graph=graph, edge=[h, patient_id], node_type = [sdata.hcp_type[h], 'patient'], 
                            #            edge_type = "transmission", edge_label= 'hcp->patient', contact_loc = cur_room, ts_start = contact_timestamp[0], ts_end = contact_timestamp[1])
                # Here calculate the HCP-HCP transmission inside of the room
                # Find who are uninfected and what is the transmission probability
                tmp_infection_list = dict()
                
                for index1 in overlapped_contact_index:
                    index_start1 = sdata.hcp_visits[index1][2] #2->start_time
                    index_end1 = sdata.hcp_visits[index1][3] #3->end_time
                    mapped_hcp1 = overlapped_index_to_hcp_mapping[index1]
                    if mapped_hcp1 == None:
                        continue
                    mapped_hcp1_trans_prob = 1
                    infected_by = list()

                    for index2 in overlapped_contact_index:
                        index_start2 = sdata.hcp_visits[index2][2] #2->start_time
                        index_end2 = sdata.hcp_visits[index2][3] #3->end_time
                        mapped_hcp2 = overlapped_index_to_hcp_mapping[index2]
                        if mapped_hcp2 == mapped_hcp1 or mapped_hcp2 == None:
                            continue

                        # Overlapped time is the contact duration
                        # Overlapped time = (latest_start, earliest_end)
                        ts_start = max(index_start1, index_start2)
                        ts_end = min(index_end1, index_end2)

                        if mapped_hcp2 not in infection_list or (mapped_hcp2 in infection_list and infection_list[mapped_hcp2].status!=STATUS_SHEDDING):
                                continue
 
                        # who infects and for how long it contacts
                        # find overlapping time between mapped_hcp1 and mapped_hcp2
                        # calculate prob based on the overlapped time
                        # overlapped_time in seconds
                        overlapped_time = find_overlapping_period(startA=index_start1, endA = index_end1,
                                                                        startB = index_start2, endB = index_end2)
                        shedding = shedding_dic[model_type][infection_list[mapped_hcp2].inf_day]/shedding_scale
                        shedding*=sdata.trans_prob
                        scaled_shedding = 1 - math.pow((1 - shedding), math.ceil(overlapped_time / 30)) # 30 seconds
                        mapped_hcp1_trans_prob*=(1-scaled_shedding)
                        infected_by.append([mapped_hcp2, [ts_start, ts_end] ])

                    mapped_hcp1_trans_prob = (1-mapped_hcp1_trans_prob)
                    if mapped_hcp1 not in infection_list and random.random() < mapped_hcp1_trans_prob:
                        # HCP get infected from HCPs
                        infectionData = InfectionData(inf_day=0,is_first_infected = False, infected_by="hcp", trans_prob=sdata.trans_prob, infected_at=cur_room, status=STATUS_INFECTED)
                            
                        #infection_list[mapped_hcp1] = infectionData
                        for h, contact_timestamp in infected_by:
                            infection_list[h].secondary_infs.append(mapped_hcp1)
                            
                        tmp_infection_list[mapped_hcp1] = infectionData
                        if args.simulation != 'base':
                            if cur_room not in sdata.b_assignment[src_bubble]:
                                print("three")
                                ext_leave_cnt+=1
                                ext_reach_cnt+=1
                            else:
                                if mapped_hcp1 not in sdata.hcp_bubble: # non-nurse
                                    ext_leave_cnt+=1
                                elif sdata.hcp_bublle[mapped_hcp1] != src_bubble:
                                    ext_leave_cnt+=1
            
                for key in tmp_infection_list:
                    infection_list[key] = tmp_infection_list[key]
                # update the last visits of overlapped HCPs
                for index in overlapped_contact_index:
                    mapped_hcp = overlapped_index_to_hcp_mapping[index]
                    if mapped_hcp == None:
                        continue
                    index_start = sdata.hcp_visits[index][2] #2->start_time
                    index_end = sdata.hcp_visits[index][3] #3->end_time
                    index_normalized_end = get_normalized_time(index_end, sdata.interaction_date,day)
                    index_normalized_start = get_normalized_time(index_start, sdata.interaction_date,day)
                       
                    last_visit[mapped_hcp]['start'] = index_normalized_start
                    last_visit[mapped_hcp]['end'] = index_normalized_end
                    last_visit[mapped_hcp]['room'] = cur_room 

        for key in infection_list.keys():
            infection_list[key].inf_day+=1


    #if args.save_transmission_pathways=="yes":
    #    store_transmission_graph(graph,sdata)

    # Save once
    if sdata.replicate_id == 0:
        if args.simulation != 'base':
            store_rewired_network(args = args, adj=adj, n_bubble=args.n_bubble, node_list=node_list,
                              simulation= args.simulation)
        if args.simulation == 'base':
            store_baseline_load_demand_mobility(args = args, load_list= load_list, demand_list=demand_list, mobility_list = mobility_list)
            pathlib.Path("dumps/mobility/").mkdir(parents=True, exist_ok=True)
            dump_load_pickle_object("dump", filename="dumps/mobility/"+args.data+"/baseline_visit_graph",data=visit_history)
    if args.simulation != 'base':
        print(ext_leave_cnt, ext_reach_cnt)
    return infection_list, load_list, demand_list, mobility_list


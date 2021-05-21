import argparse

from pandas.io.formats.format import Datetime64TZFormatter
from covid19_simulation.micu_covid import MICUCovid19Simulation
from covid19_simulation.ltcf_covid import LTCFCovid19Simulation
from datetime import datetime
from pathlib import Path
import json

from utils import *
from bubble_clustering.micu import *
from bubble_clustering.ltcf import *


parser = argparse.ArgumentParser(description="CoRN")
parser.add_argument('--data',type=str, default='MICU', help='Switch to different datasets: MICU, LTCF_small, LTCF_large')
parser.add_argument('--n_bubble', type=int, default=5, help="Choose the number of bubbles")

# Disease simulation setup
parser.add_argument('--simulation', type=str, default='base', help='Select the simulation type: base/rewired_ilp/rewired_random')
parser.add_argument('--n_replicate', type=int, default=500, help="Number of replicates to run the simulation")
parser.add_argument('--inf_source', type=str, default='nurse', help="Infection source at the beginning of the simulation; chosse patient/nurse/non-nurse")
parser.add_argument('--trans_prob', type=float, default=0.0015, help = "Tranmission probability for the covid19 simulation")

# ILP setup
parser.add_argument('--D', type=float, default=50, help='Bubble diameter (Upper bound)')
parser.add_argument('--Y', type=float, default=100, help='Excess load upper bound')

# Experiment setup
parser.add_argument('--n_cpu',type=int, default=40, help="Number of cpu's to use for multiprocessing")
args = parser.parse_args()


def create_bubbles(bble_clustering):
    random_clustering = [] # Each replicate has different random clustering
    ilp_clustering = [] # Each replicate runs over the same best optimal partition
    for i in range(args.n_replicate):
        h_bubble, h_per_room, assignment = bble_clustering.create_random_bubbles()
        random_clustering.append({ 'h_bubble':h_bubble, 
                                    'h_per_room':h_per_room, 
                                    'assignment': assignment})

    h_bubble, h_per_room, clustering = bble_clustering.create_ILP_bubbles()
    if len(clustering.keys()) != 0: # feasible clustering
        for i in range(args.n_replicate):
            ilp_clustering.append({ 'h_bubble':h_bubble, 
                                'h_per_room':h_per_room, 
                                'assignment': assignment})
    return random_clustering, ilp_clustering



def start_simulation(simulator, random_clustering, ilp_clustering):
    # Baseline simulation; no rewiring
    args.simulation = 'base'
    args.clustering = None
    res_base = simulator.run_covid19_simulation()

    # Rewired simulation;
    # Healthcare professionals and patients rooms 
    # are randly clustered into bubbles;
    # No cost optimization;
    args.simulation = 'rewired_random'
    args.clustering = random_clustering
    res_random = simulator.run_covid19_simulation()

    # Rewired simulation with cost optimization;
    # Optimal clustering of HCPs and patient rooms into bubbles 
    args.simulation = 'rewired_ilp'
    args.clustering = ilp_clustering 
    res_ilp = simulator.run_covid19_simulation()

    # Store the simulation results
    store_results(res_base, res_random, res_ilp)

    return res_base, res_random, res_ilp



def store_results(res_base, res_random, res_ilp):
    # Write a custom np_encoder for writing json with int64 type value
    # Otherwise, convert all writable items to python int
    def np_encoder(object):
        if isinstance(object, (np.generic, np.ndarray)):
            return object.item()
    # Save bubbles 
    dic = {}
    dic['Transmission probability'] = args.trans_prob
    dic['Number of bubbles'] = args.n_bubble
    dic["Param 'D'"] = args.D 
    dic["Param 'Y'"] = args.Y 
    dic['Baseline'] = res_base
    dic['Rewired-random'] = res_random
    dic['Rewired-ilp'] = res_ilp
    Path("output/results/"+args.data+"/").mkdir(parents=True, exist_ok=True)
    file = "output/results/"+args.data+"/"+"simulation_results.pickle"

    with open(file, 'wb') as f:
        pickle.dump(dic, f)
        #json.dump(dic, f, sort_keys=True, indent=4, default=np_encoder)



def main():
    # The datasets are: MICU and LTCF (small, large)
    # There are some differences between MICU and LTCF data
    # Separately handle the experiment for each dataset
    # 
    if args.data == 'MICU':
        h_visits, p_stay, h_list, r_list, h_type, r_dists = load_data(args)
   
        # Do Bubble clustering on HCPs and Patient rooms based on baseline mobility graph
        # Two types of clustering techniques:
        # 1. Random clustering: 
        # HCPs and rooms are randomly partitioned among the bubbles; does not consider any cost
        # each replicate may generate different random clustering
        #
        # 2. Integer Linear Program (ILP) clustering: 
        # partitions HCPs and rooms into different bubbles by reducing the different cost
        bble_clustering = MICUBubbleClustering(args, h_list, r_list, h_type, r_dists, h_visits, p_stay)
        random_clustering, ilp_clustering = create_bubbles(bble_clustering)
        if len(ilp_clustering)==0:
            print("Integer Linear Program: No feasible soultion")
            return


        # Run the COVID-19 simulation of three kinds
        # 1. Baseline: run the simulation on baseline HCP visits
        # 2. Rewired random: rewire the HCP visits within the bubble constructed randomly
        # 3. Rewired ILP: rewire the visit edges within the bubble constructed optimally by solving the linear program

        simulator = MICUCovid19Simulation(args, h_visits, p_stay, h_list, r_list, h_type, r_dists)
        res_base, res_random, res_ilp = start_simulation(simulator, random_clustering, ilp_clustering)




    # LTCF data are of two types: small and large
    # Both of them have similar structure and can be used interchangeably
    elif args.data == "LTCF_small" or args.data == "LTCF_large":
        h_visits, h_list, r_list, h_type, r_type, r_dists = load_data(args)

        # Run the Bubble Clustering
        bble_clustering = LTCFBubbleClustering(args, h_list, r_list, h_type, r_type, r_dists, h_visits)
        random_clustering, ilp_clustering = create_bubbles(bble_clustering)
        if len(ilp_clustering)==0:
            print("Integer Linear Program: No feasible soultion")
            return

        # Run the COVID-19 simulation of three kinds
        simulator = LTCFCovid19Simulation(args, h_visits, h_list, r_list, h_type, r_type, r_dists)
        res_base, res_random, res_ilp = start_simulation(simulator, random_clustering, ilp_clustering)


if __name__ == "__main__":
    main()
    
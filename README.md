# Cost-aware Rewiring of Network (CoRN)
This repository contains the implementation of Cost-aware Rewiring of Network for healthcare bubbles as desribed in our paper: <br\>
["Modeling and Evaluation of Clustering Patient Care into Bubbles"](https://arxiv.org/abs/2105.06278) (IEEE-ICHI 2021). 

### Requirements
- numpy
- gurobipy (for linear program optimization)


### Run the demo
```python main.py --data MICU --n_replicate 100 --n_cpu 20 --n_bubble 3 --trans_prob 0.001 --D 5 --Y 10```

### Data
In order to user your own data, you have to provide the followings:
- Healtcare professional's (HCPs) visit data: ```[hcp_id, patient_id, visit_start_time, visit_end_time]```
- A list of HCPs: ```[hcp_id_1, hcp_id_2,..., hcp_id_N]```
- A list of rooms: ```[room_id_1, room_id_2,..., room_id_M ]```
- A mapping of HCP type: ```{hcp_id_1: 'nurse', hcp_id_2: 'doctor',...,hcp_id_N: 'staff'}```
- A M by M distance matrix (M is the number of rooms)

Have a look at the ```load_data()``` function in ```utils.py``` for an example. <br\>
In this example, we load three different sets of data: Medical Intensive Care Unit (MICU) data, and two Long-Term Care Facility (LTCF) data.   

### Cite
Please cite our paper if you use this code or our dataset in your own work:

```
@misc{hasan2021modeling,
      title={Modeling and Evaluation of Clustering Patient Care into Bubbles}, 
      author={D. M. Hasibul Hasan and Alex Rohwer and Hankyu Jang and Ted Herman and Philip M. Polgreen and Daniel K. Sewell and Bijaya Adhikari and Sriram V. Pemmaraju},
      year={2021},
      eprint={2105.06278},
      archivePrefix={arXiv},
      primaryClass={cs.SI}
}

```
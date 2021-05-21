# Cost-aware Rewiring of Network (CoRN)
This repository contains the implementation of Cost-aware Rewiring of Network for healthcare bubbles as desribed in our paper: <br/> <br/>
Hasan et al. ["Modeling and Evaluation of Clustering Patient Care into Bubbles"](https://arxiv.org/abs/2105.06278) (To appear in IEEE-ICHI 2021). 

### Requirements
- numpy
- gurobipy (for linear program optimization)


### Run the demo
```python main.py --data MICU --n_replicate 500 --n_bubble 3```

### Data
In order to use your own data, you have to provide the followings:
- Timestamped mobility data: ```[HCP_ID, ROOM_ID, START_TS, END_TS]```
- A list of HCPs: ```[HCP_1, HCP_2,..., HCP_N]```
- A list of rooms: ```[ROOM_1, ROOM_2,..., ROOM_M ]```
- A mapping of HCP type: ```{HCP_1: 'nurse', HCP_2: 'doctor',...,HCP_N: 'staff'}```
- A M by M distance matrix (M is the number of rooms)

Have a look at the ```load_data()``` function in ```utils.py``` for an example. <br/>
In this example, we load three different sets of data: Medical Intensive Care Unit (MICU) data, and two Long-Term Care Facility (LTCF) data. <br/>
The MICU mobility data looks like below:
| Visit   | HCP_ID  | ROOM_ID  | START              |           END      | 
| :-----: | :-:     | :-:      | :-:                | :-:                |
| 1       | 3       | 17503    |2011-06-03 07:09:03 |2011-06-03 07:28:54 |
| 2       | 15      | 17413    |2011-06-03 07:15:03 |2011-06-03 07:16:06 |
| 3       | 7       | 17438    |2011-06-03 07:17:01 |2011-06-03 07:19:41 |

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

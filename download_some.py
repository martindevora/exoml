import os.path

import pandas as pd
from sklearn.utils import shuffle


local_path = "/mnt/DATA/ml/"
if not os.path.exists(local_path):
    os.mkdir(local_path)
local_path_ete6 = f'{local_path}ete6/'
if not os.path.exists(local_path_ete6):
    os.mkdir(local_path_ete6)
    os.mkdir(local_path_ete6 + '/lcs/')
local_path_q1q17 = f'{local_path}q1q17/'
if not os.path.exists(local_path_q1q17):
    os.mkdir(local_path_q1q17)
    os.mkdir(local_path_q1q17 + '/lcs/')
ete6_local_file = f"{local_path_ete6}injected_objects.csv"
q1q17_local_file = f"{local_path_q1q17}injected_objects_tces.csv"
user = "user"
pwd = 'fakepwd'
result = os.system(f"sshpass -p {pwd}. scp -o StrictHostKeyChecking=no {user}@mdevorapajares.net:/data/scratch/ml/ete6/injected_objects.csv {ete6_local_file}")
result = os.system(f"sshpass -p {pwd}. scp -o StrictHostKeyChecking=no {user}@mdevorapajares.net:/data/scratch/ml/q1_q17/injected_objects_tces.csv {q1q17_local_file}")
df_ete6 = pd.read_csv(ete6_local_file)
#df_q1q17 = pd.read_csv(q1q17_local_file)
df = pd.concat([df_ete6])
df = shuffle(df)
df = df.sort_values(by=["object_id"], ascending=True)
df = df.drop_duplicates(subset=["object_id"], keep='last')
df = df.iloc[0:1000]
df.to_csv(local_path + '/injected_objects_combined.csv')
for index, row in df.iterrows():
    remotepath = 'ete6/lcs/' if 'TIC' in row['object_id'] else 'q1_q17/lcs/'
    local_path_target = f'{local_path_ete6}lcs/' if 'TIC' in row['object_id'] else f'{local_path_q1q17}lcs/'
    object_path = '_'.join(row['object_id'].split(' ')) + '_' + str(round(row['period'], 2))
    print(f"Downloading {object_path}")
    result = os.system(f"sshpass -p {pwd} scp -o StrictHostKeyChecking=no {user}@mdevorapajares.net:/data/scratch/ml/{remotepath}{object_path}_lc_short.csv {local_path_target}{object_path}_lc_short.csv")
    result = os.system(f"sshpass -p {pwd} scp -o StrictHostKeyChecking=no {user}@mdevorapajares.net:/data/scratch/ml/{remotepath}{object_path}_lc_long.csv {local_path_target}{object_path}_lc_long.csv")

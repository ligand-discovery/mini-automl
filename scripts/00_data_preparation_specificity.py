import os
import pandas as pd
import numpy as np
import collections

root = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(root, "..", "data")

df = pd.read_csv(os.path.join(data_dir, "screening.tsv"), sep="\t")
db = pd.read_csv(os.path.join(data_dir, "screening_hits.tsv"), sep="\t")
fs = pd.read_csv(os.path.join(data_dir, "fid2can_fff_all.tsv"), sep="\t")
fid2smi = dict((r[0], r[1]) for r in fs.values)

fids = sorted(set(df["FragID"]))
smiles_list = [fid2smi[fid] for fid in fids]

fid2pid = collections.defaultdict(list)
pid2fid = collections.defaultdict(list)
for v in db[["Accession", "FragID"]].values:
    pid2fid[v[0]] += [v[1]]
    fid2pid[v[1]] += [v[0]]

pid2cat = collections.defaultdict(list)
for k,v in pid2fid.items():
    if len(v) < 4:
        pid2cat[0] += [k]
        continue
    if len(v) >= 40:
        pid2cat[2] += [k]
        continue
    pid2cat[1] += [k]
pid2cat = dict((k, set(v)) for k,v in pid2cat.items())

r = []
for k,v in fid2pid.items():
    r += [len(pid2cat[2].intersection(v))]

cuts = {
    0: [5, 10, 20],
    1: [10, 50, 100],
    2: [50, 100, 200]
}

slots = []
for i in range(3):
    for j in range(3):
        slots += [(i,j)]

slot2fid = collections.defaultdict(list)
for slot in slots:
    pids_ = pid2cat[slot[0]]
    for k,v in fid2pid.items():
        common = pids_.intersection(v)
        n = cuts[slot[0]][slot[1]]
        if len(common) >= n:
            slot2fid[slot] += [k]

slot2fid = dict((k, set(v)) for k,v in slot2fid.items())

R = []
for s in slots:
    fids_ = slot2fid[s]
    y_ = []
    for fid in fids:
        if fid in fids_:
            y_ += [1]
        else:
            y_ += [0]
    R += [y_]

Y = np.array(R).T

R = []
for i in range(Y.shape[0]):
    r = list(Y[i,:])
    R += [[fids[i], smiles_list[i]] + r]
    
do = pd.DataFrame(R, columns = ["fid", "smiles"] + ["sensitivity_fxp{0}_pxf{1}".format(x[0], x[1]) for x in slots])

fid2pid_counts = dict((k, len(v)) for k,v in fid2pid.items())
fid2pid_counts = sorted(fid2pid_counts.items(), key=lambda x: -x[1])

idxs = [i for i, fid in enumerate(fids) if len(fid2pid[fid]) > 0]
fids = [fids[i] for i in idxs]
smiles_list = [smiles_list[i] for i in idxs]

cuts_pxf = [5, 20, 50]

single_cuts = collections.defaultdict(list)
for i, c in enumerate(cuts_pxf):
    for fid in fids:
        pids_ = fid2pid[fid]
        if len(pids_) <= c:
            single_cuts[i] += [fid]
single_cuts = dict((k, set(v)) for k,v in single_cuts.items())

R = []
for i, c in enumerate(cuts_pxf):
    fids_ = single_cuts[i]
    y_ = []
    for fid in fids:
        if fid in fids_:
            y_ += [1]
        else:
            y_ += [0]
    R += [y_]

Y = np.array(R).T
print(np.sum(Y, axis=0))

R = []
for i in range(Y.shape[0]):
    r = list(Y[i,:])
    R += [[fids[i], smiles_list[i]] + r]
    
dp = pd.DataFrame(R, columns = ["fid", "smiles"] + ["specificity_pxf{0}".format(i) for i in range(len(cuts_pxf))])

dp.to_csv(os.path.join(data_dir, "specificity_pxf.csv"), index=False)
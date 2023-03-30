import argparse
import json
import os
import shutil
import scipy.io
import torch
import numpy as np


#######################################################################
# Evaluate
def evaluate(qf, ql, qc, gf, gl, gc, filenames):
    # ==== DEBUG ====
    debug_dir = f"{args.debug_dir}/{ql}c{qc}"
    if not os.path.exists(debug_dir):
        os.mkdir(debug_dir)
    # ===============

    query = qf.view(-1, 1)

    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]
    # good index
    query_index = np.argwhere(gl == ql)
    camera_index = np.argwhere(gc == qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl < 0)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)

    CMC_tmp = compute_mAP(index, good_index, junk_index, filenames, debug_dir)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index, filenames, debug_dir):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask)
    rows_good = rows_good.flatten()

    # ===== DEBUG =====
    for i, idx in enumerate(index[:10]):
        shutil.copy(
            filenames[str(idx)],
            f"{debug_dir}/{i}r{rows_good[0]}_{filenames[str(idx)].split('/')[-1]}",
        )
    # =================

    cmc[rows_good[0] :] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc


######################################################################
result = scipy.io.loadmat("pytorch_result.mat")
query_feature = torch.FloatTensor(result["query_f"])
query_cam = result["query_cam"][0]
query_label = result["query_label"][0]
gallery_feature = torch.FloatTensor(result["gallery_f"])
gallery_cam = result["gallery_cam"][0]
gallery_label = result["gallery_label"][0]

multi = os.path.isfile("multi_query.mat")

if multi:
    m_result = scipy.io.loadmat("multi_query.mat")
    mquery_feature = torch.FloatTensor(m_result["mquery_f"])
    mquery_cam = m_result["mquery_cam"][0]
    mquery_label = m_result["mquery_label"][0]
    mquery_feature = mquery_feature.cuda()

parser = argparse.ArgumentParser(description="Evaluate gpu")
parser.add_argument("debug_dir", help="path to debug dir")
args = parser.parse_args()

# ====== DEBUG ======
with open(f"{args.debug_dir}/filenames.json", "r", encoding="utf-8") as rfile:
    filenames = json.load(rfile)
# ===================

query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()

print(query_feature.shape)
CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0
# print(query_label)
for i, _ in enumerate(query_label):
    ap_tmp, CMC_tmp = evaluate(
        query_feature[i],
        query_label[i],
        query_cam[i],
        gallery_feature,
        gallery_label,
        gallery_cam,
        filenames,
    )
    # ==== DEBUG ====
    # print(f"{query_label[i]}{query_cam[i]}: {CMC_tmp[0]}")
    if CMC_tmp[0] == -1:
        continue
    CMC = CMC + CMC_tmp
    ap += ap_tmp
    # print(i, CMC_tmp[0])

CMC = CMC.float()
CMC = CMC / len(query_label)  # average CMC
print(
    "Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f"
    % (CMC[0], CMC[4], CMC[9], ap / len(query_label))
)

# multiple-query
CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0
if multi:
    for i in range(len(query_label)):
        mquery_index1 = np.argwhere(mquery_label == query_label[i])
        mquery_index2 = np.argwhere(mquery_cam == query_cam[i])
        mquery_index = np.intersect1d(mquery_index1, mquery_index2)
        mq = torch.mean(mquery_feature[mquery_index, :], dim=0)
        ap_tmp, CMC_tmp = evaluate(
            mq,
            query_label[i],
            query_cam[i],
            gallery_feature,
            gallery_label,
            gallery_cam,
            filenames,
        )
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
        # print(i, CMC_tmp[0])
    CMC = CMC.float()
    CMC = CMC / len(query_label)  # average CMC
    print(
        "multi Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f"
        % (CMC[0], CMC[4], CMC[9], ap / len(query_label))
    )

import numpy as np

from logger import log_eval


def compute_ap(ranks, nres):
    """
    Computes average precision for given ranked indexes.
    
    Arguments
    ---------
    ranks : zerro-based ranks of positive images
    nres  : number of positive images
    
    Returns
    -------
    ap    : average precision
    """

    # number of images ranked by the system
    nimgranks = len(ranks)

    # accumulate trapezoids in PR-plot
    ap = 0

    recall_step = 1. / nres

    for j in np.arange(nimgranks):
        rank = ranks[j]

        if rank == 0:
            precision_0 = 1.
        else:
            precision_0 = float(j) / rank

        precision_1 = float(j + 1) / (rank + 1)

        ap += (precision_0 + precision_1) * recall_step / 2.

    return ap

def compute_map(ranks, gnd, kappas=[]):
    """
    Computes the mAP for a given set of returned results.

         Usage: 
           map = compute_map (ranks, gnd) 
                 computes mean average precsion (map) only
        
           map, aps, pr, prs = compute_map (ranks, gnd, kappas) 
                 computes mean average precision (map), average precision (aps) for each query
                 computes mean precision at kappas (pr), precision at kappas (prs) for each query
        
         Notes:
         1) ranks starts from 0, ranks.shape = db_size X #queries
         2) The junk results (e.g., the query itself) should be declared in the gnd stuct array
         3) If there are no positive images for some query, that query is excluded from the evaluation
    """

    map = 0.
    nq = len(gnd)  # number of queries
    aps = np.zeros(nq)
    pr = np.zeros(len(kappas))
    prs = np.zeros((nq, len(kappas)))
    rec = np.zeros(len(kappas))
    apk = np.zeros(len(kappas))
    nempty = 0

    for i in np.arange(nq):
        qgnd = np.array(gnd[i]['ok'])

        # no positive images, skip from the average
        if qgnd.shape[0] == 0:
            aps[i] = float('nan')
            prs[i, :] = float('nan')
            nempty += 1
            continue

        try:
            qgndj = np.array(gnd[i]['junk'])
        except:
            qgndj = np.empty(0)

        # sorted positions of positive and junk images (0 based)
        pos = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], qgnd)]
        junk = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], qgndj)]

        k = 0
        ij = 0
        if len(junk):
            # decrease positions of positives based on the number of
            # junk images appearing before them
            ip = 0
            while (ip < len(pos)):
                while (ij < len(junk) and pos[ip] > junk[ij]):
                    k += 1
                    ij += 1
                pos[ip] = pos[ip] - k
                ip += 1

        # compute ap
        ap = compute_ap(pos, len(qgnd))
        map = map + ap
        aps[i] = ap

        # compute map @ k
        for j in np.arange(len(kappas)):
            ak_pos = pos[pos <= kappas[j] - 1]
            apk[j] += compute_ap(ak_pos, min(len(qgnd), kappas[j]))
            rec[j] += ak_pos.shape[0] > 0

        # compute precision @ k
        pos += 1  # get it to 1-based
        for j in np.arange(len(kappas)):
            kq = min(max(pos), kappas[j])
            prs[i, j] = (pos <= kq).sum() / kq
        pr = pr + prs[i, :]

    map = map / (nq - nempty)
    pr = pr / (nq - nempty)
    apk = apk / (nq - nempty)
    rec = rec / (nq - nempty)

    return map, aps, pr, prs, apk, rec


def compute_map_and_print(dataset, ranks, gnd, logger=None, kappas=[1, 4, 8]):
    map, aps, mpr, prs, apk, rec = compute_map(ranks, gnd, kappas)
    res_1 = '>> {}: mAP {}'.format(dataset, np.around(map * 100, decimals=2))
    res_2 = '>> {}: mP@k{} {}'.format(dataset, kappas, np.around(mpr * 100, decimals=2))
    res_3 = '>> {}: mAP@k{} {}'.format(dataset, kappas, np.around(apk * 100, decimals=2))
    res_4 = '>> {}: R@k{} {}'.format(dataset, kappas, np.around(rec * 100, decimals=2))
    print(res_1)
    print(res_2)
    print(res_3)
    print(res_4)
    log_eval(logger, {f'test/map_{dataset}': map}, res_1)
    log_eval(logger, {f'test/p@1_{dataset}': rec[0]}, res_4)
    return map

def compute_sim_and_print(dataset, vecs, qvecs, logger=None):
    sim = np.power((vecs * qvecs).sum(0), 2).mean()
    res = '>> {}: sim asym. sq {}'.format(dataset, np.around(sim * 100, decimals=2))
    print(res)
    log_eval(logger, {f'test/sim_asym_{dataset}': sim}, res)
    return sim
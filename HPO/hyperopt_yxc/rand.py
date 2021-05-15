"""
Random search - presented as hyperopt.fmin_random
"""
from __future__ import absolute_import
import logging
import numpy as np

from . import pyll

from .base import miscs_update_idxs_vals

logger = logging.getLogger(__name__)

#rng用法:  得到相同的随机数组
#rng = np.random.RandomState(0)
#rng.rand(4)
#Out[377]: array([0.5488135 , 0.71518937, 0.60276338, 0.54488318])
def suggest(new_ids, domain, trials, seed):
    rng = np.random.RandomState(seed)
    rval = []
    for ii, new_id in enumerate(new_ids):
        # -- sample new specs, idxs, vals
        idxs, vals = pyll.rec_eval(
            domain.s_idxs_vals,
            memo={
                domain.s_new_ids: [new_id],
                domain.s_rng: rng,#domain：域  超参数的域 specifies a search problem指定一个搜索问题
            })
        new_result = domain.new_result()
        new_misc = dict(tid=new_id, cmd=domain.cmd, workdir=domain.workdir)
        miscs_update_idxs_vals([new_misc], idxs, vals)#更新idxs, vals
        rval.extend(trials.new_trial_docs([new_id],
                    [None], [new_result], [new_misc]))#doc['book_time']=None
    return rval


def suggest_batch(new_ids, domain, trials, seed):

    rng = np.random.RandomState(seed)
    # -- sample new specs, idxs, vals
    idxs, vals = pyll.rec_eval(
        domain.s_idxs_vals,
        memo={
            domain.s_new_ids: new_ids,
            domain.s_rng: rng,
        })
    return idxs, vals


# flake8 likes no trailing blank line

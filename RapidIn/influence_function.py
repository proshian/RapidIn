#! /usr/bin/env python3

import time

from RapidIn.calc_inner import s_test

IGNORE_INDEX = -100


def calc_s_test_single(model, z_test, t_test, input_len, train_loader, gpu=-1,
                       damp: float = 0.01, scale: int = 25, 
                       recursion_depth: int = 5000, r: int = 1, 
                       need_reshape: bool =True):
    """
    Calculate the s_test vector (iHVP) for a single test example using the LiSSA algorithm.
    
    Runs `s_test` function `r` times and averages the results
    to reduce variance of the stochastic estimation.
    """
    min_nan_depth = recursion_depth
    res, nan_depth = s_test(z_test, t_test, input_len, model, train_loader,
                 gpu=gpu, damp=damp, scale=scale,
                 recursion_depth=recursion_depth,
                 need_reshape=need_reshape)
    min_nan_depth = min(min_nan_depth, nan_depth)
    for i in range(1, r):
        start_time = time.time()
        cur, nan_depth = s_test(z_test, t_test, input_len, model, train_loader,
               gpu=gpu, damp=damp, scale=scale,
               recursion_depth=recursion_depth,
               need_reshape=need_reshape)
        res = res + cur
        min_nan_depth = min(min_nan_depth, nan_depth)

    if min_nan_depth != recursion_depth:
        print(f"Warning: get Nan value after depth {min_nan_depth}, current recursion_depth = {min_nan_depth}")
    res = res/r

    return res

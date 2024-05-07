import pandas as pd
from scipy import stats
import numpy as np
from pyspark.sql import SparkSession
from datetime import date, timedelta, datetime
from pyspark.sql.functions import *
from pyspark.sql.types import *
import sys
import math

spark = SparkSession.builder.appName('ab_test_split_sample').getOrCreate()
spark.sparkContext.setLogLevel('ERROR')

# define variables
# với trường hợp của AB test với tỷ lệ chia => tính sample size cho z test
def cal_min_sample_size(bcr, mde, split_ratio, alternative, alpha = 0.05, power = 0.8,):
    from statsmodels.stats.power import NormalIndPower as power
    """
    bcr: conversion rate cơ bản của tập control
    mde: kỳ vọng (variance - control)/control
    ratio: [percentage sample1, pct sample2]
    return: number of observation sample 1
    """
    effect_size = mde*bcr/math.sqrt(bcr*(1-bcr))
    n_sample1 = power().solve_power(effect_size = effect_size, alpha = 0.05, power = 0.8, ratio = split_ratio[1]/split_ratio[0], alternative = alternative)
    n_sample2 = (n_sample1*split_ratio[1])/split_ratio[0]
    return {
        "n_sample1" : n_sample1,
        "n_sample2" :n_sample2,
        "total_sample": n_sample1 + n_sample2
    }

# với trường hợp sample size cho N variance version => tính sample size để đảm bảo các version variance có phân phối giống nhau
def calculate_sample_size_abn(bcr, mde, power, alpha, num_groups):
    """
    num groups: số groups 
    """
    from statsmodels.stats.power import GofChisquarePower as Power
    effect_size = mde*bcr/math.sqrt(bcr*(1-bcr))
    analysis = Power().solve_power(effect_size=effect_size, nobs=None, power=power, alpha=alpha, n_bins=num_groups)
    return analysis

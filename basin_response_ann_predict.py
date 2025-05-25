#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 15:03:39 2025

@author: sayan
"""

import xarray as xr
import numpy as np
import torch
import torch.nn as nn
from typing import List
import pandas as pd
import geopandas as gpd
import os
import logging

logger = logging.getLogger(__name__)

########################## Utility Functions ##########################
def mnth_select(da: xr.DataArray, month_range: List[int]) -> xr.DataArray:
    mm1, mm2 = month_range
    if mm1 <= mm2:
        return da.sel(time=da['time'].dt.month.isin(range(mm1, mm2 + 1)))
    else:
        return da.sel(time=da['time'].dt.month.isin(list(range(mm1, 13)) + list(range(1, mm2 + 1))))

########################## Model Definition ##########################
class Disch(nn.Module):
    def __init__(self, hidden_dim, activation='logsigmoid'):
        super(Disch, self).__init__()
        self.hidden1 = nn.Linear(3, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        self.act1 = self._get_activation(activation)
        self.output_act = nn.Identity()  # Can also be parameterized if needed

    def _get_activation(self, name):
        name = name.lower()
        if name == 'relu':
            return nn.ReLU()
        elif name == 'tanh':
            return nn.Tanh()
        elif name == 'sigmoid':
            return nn.Sigmoid()
        elif name == 'logsigmoid':
            return nn.LogSigmoid()
        elif name == 'leakyrelu':
            return nn.LeakyReLU()
        elif name == 'gelu':
            return nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {name}")

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        return self.output_act(self.output(x))

########################## Preprocessing ##########################
def preprocess_data(ds_x_file, ds_z_file, shp_path, tm_pr,
                    mnths,out_folder='./out',
                    stat_path='./out'):
    ds_x=xr.open_dataset(ds_x_file)
    ds_z=xr.open_dataset(ds_z_file)
    mnthsx1 = mnths
    mnthsz = mnths
    shp = gpd.read_file(shp_path)
    tsx = ds_x[list(ds_x.data_vars)[0]].rio.write_crs(str(shp.crs))
    tsx = tsx.rio.clip(shp.geometry, shp.crs, drop=False)
    tsz = ds_z[list(ds_z.data_vars)[0]].rio.write_crs(str(shp.crs))
    tsz = tsz.rio.clip(shp.geometry, shp.crs, drop=False)
    freq_x=(tsx.time.diff('time') / np.timedelta64(1, 'D')).mean().item()
    freq_z=(tsz.time.diff('time') / np.timedelta64(1, 'D')).mean().item()
    logger.info("freq_rain=%s, freq_temperature=%s", freq_x, freq_z)
    if freq_x<27:
        logger.info('Precipitation field resampled to monthly')
        tsx = tsx.resample(time='ME').mean()
    elif freq_x>31:
        raise ValueError(f"Unsupported time frequency: {freq_x:.2f} days between steps.")
    if freq_z<27:
        logger.info('Temperature field resampled to monthly')
        tsz = tsz.resample(time='ME').mean()
    elif freq_z>31:
        raise ValueError(f"Unsupported time frequency: {freq_z:.2f} days between steps.")

    tsx = tsx.weighted(np.cos(np.deg2rad(tsx.latitude))).mean(('latitude', 'longitude'))
    tsz = tsz.weighted(np.cos(np.deg2rad(tsz.latitude))).mean(('latitude', 'longitude'))

    tsx1 = mnth_select(tsx, mnthsx1)
    tsz = mnth_select(tsz, mnthsz)

    tsx1_pr = tsx1.sel(time=slice(tm_pr[0], tm_pr[1]))
    x2_pr_time = (tsx1_pr.time.to_index() - pd.DateOffset(months=1)).to_series().dt.to_period("M").dt.to_timestamp("M")
    x2_pr_time=xr.DataArray(x2_pr_time.values, dims=["time"])
    tsx2_pr = tsx.sel(time=x2_pr_time)
    tsz_pr = tsz.sel(time=slice(tm_pr[0], tm_pr[1]))
    stats=pd.read_csv(os.path.join(stat_path, 'scaling_parameter.csv'))
    x_pr= np.hstack([
        ((tsx1_pr - stats['mean'].iloc[0]) / stats['stdev'].iloc[0]).values.reshape(-1, 1),
        ((tsx2_pr -  stats['mean'].iloc[1]) / stats['stdev'].iloc[1]).values.reshape(-1, 1),
        ((tsz_pr -  stats['mean'].iloc[2]) / stats['stdev'].iloc[2]).values.reshape(-1, 1),
    ])

    time_ind_pr = tsx1_pr.time
    return x_pr,time_ind_pr
def predict(x_pr,neuron_n,time_ind_pr,
            a_func='logsigmoid',
            stat_fold='./out',out_folder='./out',
            model_state='./out/model_state.pth'):
    x_pr_tensor=torch.tensor(x_pr,dtype=torch.float64)
    model=Disch(neuron_n,a_func).double()
    model.load_state_dict(torch.load(model_state))
    model.eval()
    with torch.no_grad():
        y_pr = model(x_pr_tensor).numpy()
    stat=pd.read_csv(os.path.join(stat_fold,'scaling_parameter.csv'))
    y_pr=(y_pr*stat['stdev'].iloc[-1])+stat['mean'].iloc[-1]
    y_pr=y_pr.flatten()
    df_pr=pd.DataFrame({'time':pd.to_datetime(time_ind_pr.values),
                        'predicted':y_pr})
    df_pr.to_csv(os.path.join(out_folder,'predicted.csv'),index=False)
    logger.info('prediction saved to '+os.path.join(out_folder,'predicted.csv'))
    return df_pr

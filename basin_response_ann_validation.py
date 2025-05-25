#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 17 15:53:39 2025

@author: sayan
"""

import xarray as xr
import numpy as np
import geopandas as gpd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
import os
from typing import List
import pandas as pd
import yaml
import logging

logger = logging.getLogger(__name__)

phi = (1 + 5 ** 0.5) / 2.0

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
def preprocess_data(ds_x_file, ds_y_file, ds_z_file, shp_path, tm_tr, mnths):
    ds_x=xr.open_dataset(ds_x_file)
    ds_y=xr.open_dataset(ds_y_file)
    ds_z=xr.open_dataset(ds_z_file)
    mnthsx1 = mnths
    mnthsz = mnths
    mnthsy = mnths

    shp = gpd.read_file(shp_path)
    tsx = ds_x[list(ds_x.data_vars)[0]].rio.write_crs(str(shp.crs))
    tsx = tsx.rio.clip(shp.geometry, shp.crs, drop=False)
    tsz = ds_z[list(ds_z.data_vars)[0]].rio.write_crs(str(shp.crs))
    tsz = tsz.rio.clip(shp.geometry, shp.crs, drop=False)
    tsy = ds_y[list(ds_y.data_vars)[0]]
    freq_x=(tsx.time.diff('time') / np.timedelta64(1, 'D')).mean().item()
    freq_z=(tsz.time.diff('time') / np.timedelta64(1, 'D')).mean().item()
    freq_y=(tsy.time.diff('time') / np.timedelta64(1, 'D')).mean().item()
    (freq_x,freq_z,freq_y)
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
    if freq_y<27:
        logger.info('Discharge field resampled to monthly')
        tsy = tsy.resample(time='ME').mean()
    elif freq_y>31:
        raise ValueError(f"Unsupported time frequency: {freq_x:.2f} days between steps.")

    tsx = tsx.weighted(np.cos(np.deg2rad(tsx.latitude))).mean(('latitude', 'longitude'))
    tsz = tsz.weighted(np.cos(np.deg2rad(tsz.latitude))).mean(('latitude', 'longitude'))

    tsx1 = mnth_select(tsx, mnthsx1)
    tsz = mnth_select(tsz, mnthsz)
    tsy = mnth_select(tsy, mnthsy)

    tsx1_tr = tsx1.sel(time=slice(tm_tr[0], tm_tr[1]))
    x2_tr_time = (tsx1_tr.time.to_index() - pd.DateOffset(months=1)).to_series().dt.to_period("M").dt.to_timestamp("M")
    x2_tr_time=xr.DataArray(x2_tr_time.values, dims=["time"])
    tsx2_tr = tsx.sel(time=x2_tr_time)
    tsz_tr = tsz.sel(time=slice(tm_tr[0], tm_tr[1]))
    tsy_tr = tsy.sel(time=slice(tm_tr[0], tm_tr[1]))

    mu1, std1 = float(tsx1_tr.mean()), float(tsx1_tr.std())
    mu2, std2 = float(tsx2_tr.mean()), float(tsx2_tr.std())
    muz, stdz = float(tsz_tr.mean()), float(tsz_tr.std())
    muy, stdy = float(tsy_tr.mean()), float(tsy_tr.std())

    x = np.hstack([
        ((tsx1_tr - mu1) / std1).values.reshape(-1, 1),
        ((tsx2_tr - mu2) / std2).values.reshape(-1, 1),
        ((tsz_tr - muz) / stdz).values.reshape(-1, 1),
        #((tsyl_tr - mul) / stdl).values.reshape(-1, 1)
    ])
    y = ((tsy_tr - muy) / stdy).values.reshape(-1, 1)

    scale_stat_y = (mu1, stdy, muy)
    time_index = tsx1_tr.time
    return x, y, scale_stat_y, time_index

########################## Training Function ##########################
def train_model(x, y, neuron_list, lr, epochs, norm_stats, time_index,a_func='logsigmoid',w_dec=None, sav=True, plot_folder="./plots",cv_folder='./out'):
    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    train_loss, val_loss, rmse_all, nse_all, pall = [], [], [], [], []
    mu1, stdy, muy = norm_stats
    os.makedirs(plot_folder, exist_ok=True)
    os.makedirs(cv_folder, exist_ok=True)
    for n in neuron_list:
        dataset = TensorDataset(x_t, y_t)
        train_size = int(len(dataset) * 0.8)
        val_size = len(dataset) - train_size
        train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(n))
        
        train_loader = DataLoader(train_set, batch_size=len(train_set))
        val_loader = DataLoader(val_set, batch_size=len(val_set))

        model = Disch(n,a_func)
        loss_fn = nn.MSELoss()
        if w_dec==None:
            optimizer=optim.Adam(model.parameters(), lr=lr)
        else:
            optimizer=optim.Adam(model.parameters(), lr=lr,weight_decay=w_dec)

        epoch_trl, epoch_vl = [], []
        for epoch in range(epochs):
            model.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_trl.append(loss.item())

            model.eval()
            with torch.no_grad():
                for xb, yb in val_loader:
                    val_pred = model(xb)
                    epoch_vl.append(loss_fn(val_pred, yb).item())

        model.eval()
        with torch.no_grad():
            pred_all = model(x_t).detach().numpy().flatten()
            pred_all = (pred_all * stdy) + muy
            pall.append(pred_all)

        target_all = ((y * stdy) + muy).reshape(-1)
        rmse = np.sqrt(np.mean((pred_all - target_all) ** 2))
        residual = np.sum((target_all - pred_all) ** 2)
        variance = np.sum((target_all - np.mean(target_all)) ** 2)
        nse = 1 - residual / variance
        
        train_loss.append(epoch_trl)
        val_loss.append(epoch_vl)
        rmse_all.append(np.round(rmse, 2))
        nse_all.append(np.round(nse, 2))

        # Plotting
        epochs_arr = np.arange(1, epochs + 1)
        fig, ax = plt.subplots(figsize=(7 * phi, 7), dpi=300)
        ax.plot(epochs_arr, epoch_trl, lw=3, color='black', label='Train loss')
        ax.plot(epochs_arr, epoch_vl, lw=2, color='red', linestyle='--', label='Validation loss')
        ax.set_title(f'no. of Neurons:{n}\nrmse:{rmse:.2f} nse:{nse:.2f}', fontsize=18)
        ax.set_ylabel("MSE", fontsize=20)
        ax.set_xlabel("epochs", fontsize=20)
        ax.tick_params(axis='both', labelsize=18, width=3, length=10)
        ax.legend(loc='best', prop={'size': 15})

        if sav:
            fig.savefig(f"{plot_folder}/nn_{n}lr_{lr}_{a_func}_wd_{w_dec}.png", bbox_inches='tight')
        else:
            plt.show()
    d_stats=pd.DataFrame({'Hidden Neuron':neuron_list,
                          'rmse':rmse_all,
                          'nse':nse_all})
    d_stats.to_csv(os.path.join(cv_folder,'cv_stats.csv'),index=False)
    logger.info('training set size='+str(train_size))
    logger.info('validation size='+str(val_size))
    logger.info(d_stats)
    return d_stats, pall, time_index

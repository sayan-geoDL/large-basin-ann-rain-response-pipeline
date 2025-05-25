#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 14:11:40 2025

@author: sayan
"""
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch
import torch.nn as nn
import torch.optim as optim
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
def preprocess_data(ds_x_file, ds_y_file, ds_z_file, shp_path, tm_tr, tm_tst, mnths,out_folder='./out'):
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
    logger.info("freq_rain=%s, freq_temperature=%s, freq_discharge=%s", freq_x, freq_z,freq_y)
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

    x_tr= np.hstack([
        ((tsx1_tr - mu1) / std1).values.reshape(-1, 1),
        ((tsx2_tr - mu2) / std2).values.reshape(-1, 1),
        ((tsz_tr - muz) / stdz).values.reshape(-1, 1),
    ])
    y_tr = ((tsy_tr - muy) / stdy).values.reshape(-1, 1)

    stats=[(mu1,std1),(mu2,std2),(muz,stdz),(muy,stdy)]
    stats=pd.DataFrame(stats,columns=['mean','stdev'])
    os.makedirs(out_folder,exist_ok=True)
    stats.to_csv(os.path.join(out_folder, 'scaling_parameter.csv'),index=False)
    time_ind_tr = tsx1_tr.time
    
    tsx1_tst = tsx1.sel(time=slice(tm_tst[0], tm_tst[1]))
    x2_tst_time = (tsx1_tst.time.to_index() - pd.DateOffset(months=1)).to_series().dt.to_period("M").dt.to_timestamp("M")
    x2_tst_time=xr.DataArray(x2_tst_time.values, dims=["time"])
    tsx2_tst = tsx.sel(time=x2_tst_time)
    tsz_tst = tsz.sel(time=slice(tm_tst[0], tm_tst[1]))
    tsy_tst = tsy.sel(time=slice(tm_tst[0], tm_tst[1]))
    
    x_tst= np.hstack([
        ((tsx1_tst - mu1) / std1).values.reshape(-1, 1),
        ((tsx2_tst - mu2) / std2).values.reshape(-1, 1),
        ((tsz_tst - muz) / stdz).values.reshape(-1, 1),
    ])
    y_tst = ((tsy_tst - muy) / stdy).values.reshape(-1, 1)
    
    time_ind_tst = tsx1_tst.time
    return x_tr, y_tr, x_tst, y_tst, stats, time_ind_tr, time_ind_tst
################ training #####################################################
def trainer(x_tr,y_tr,x_tst,y_tst,epochs,lr,neuron_n,stats,
            time_tr,time_tst,a_func='logsigmoid',w_dec=None,plot_folder="./plots",out_folder='./out'):
    phi = (1 + 5 ** 0.5) / 2.0
    model=Disch(neuron_n,a_func).double()
    x_tr_tensor=torch.tensor(x_tr,dtype=torch.float64)
    y_tr_tensor=torch.tensor(y_tr,dtype=torch.float64)
    criterion=nn.MSELoss()
    if w_dec==None:
        optimizer=optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer=optim.Adam(model.parameters(), lr=lr,weight_decay=w_dec)
    loss_history=[]
    epoch_no=[]
    for epoch in range(epochs):
        epoch_no.append(epoch+1)
        model.train()
        optimizer.zero_grad()
        outputs=model(x_tr_tensor)
        loss=criterion(outputs,y_tr_tensor)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        if (epoch+1)%20==0:
            logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")
    model.eval()
    p_tr=model(x_tr_tensor).detach().numpy()[:,0]
    p_tr=(p_tr*stats['stdev'].iloc[-1])+stats['mean'].iloc[-1]
    obs_tr = (y_tr.flatten()*stats['stdev'].iloc[-1])+stats['mean'].iloc[-1]
    nse_tr=1 - (np.sum((p_tr - obs_tr) ** 2) / np.sum((obs_tr - np.mean(obs_tr)) ** 2))
    rmse_tr=np.sqrt(((p_tr-obs_tr)**2).mean())
    fig1,ax1=plt.subplots(figsize=(7*phi,7),dpi=300)
    logger.info('nse= '+str(nse_tr))
    logger.info('rmse= '+str(rmse_tr))
    ax1.plot(epoch_no,loss_history,lw=3,label='MSE',color='black')
    ax1.set_xlabel("Epoch",fontsize=20)
    ax1.set_ylabel("MSE Loss",fontsize=20)
    ax1.tick_params(axis="both", colors="black", labelcolor="black",
                      labelsize=18,width=3,length=10)
    os.makedirs(plot_folder, exist_ok=True)
    fig1.savefig(f"{plot_folder}/train_test_loss.png", bbox_inches='tight')
    x_tst_tensor=torch.tensor(x_tst,dtype=torch.float64)
    model.eval()
    with torch.no_grad():
        p_tst=model(x_tst_tensor).detach().cpu().numpy()[:,0]
    p_tst=(p_tst*stats['stdev'].iloc[-1])+stats['mean'].iloc[-1]
    obs_tst = (y_tst.flatten()*stats['stdev'].iloc[-1])+stats['mean'].iloc[-1]
    nse_tst=1 - (np.sum((p_tst - obs_tst) ** 2) / np.sum((obs_tst - np.mean(obs_tst)) ** 2))
    rmse_tst=np.sqrt(((p_tst-obs_tst)**2).mean())
    p=np.hstack((p_tr,p_tst ))
    o=np.hstack((obs_tr,obs_tst ))
    time_all=xr.concat([time_tr,time_tst], dim='time')
    time_all=time_all.sortby('time')
    df_tr=pd.DataFrame({'time':pd.to_datetime(time_tr.values),
                        'predicted':p_tr,
                        'observed':obs_tr,})
    df_tst=pd.DataFrame({'time':pd.to_datetime(time_tst.values),
                        'predicted':p_tst,
                        'observed':obs_tst,})
    
    df_all=pd.DataFrame({'time':pd.to_datetime(time_all.values),
                        'predicted':p,
                        'observed':o,})
    fig2,ax2=plt.subplots(figsize=(7*phi,7),dpi=300)
    formatter1 = mticker.ScalarFormatter(useMathText=True)
    formatter1.set_scientific(True)
    formatter1.set_powerlimits((-3, 3))
    ax2.plot(df_tr['time'],df_tr['observed'],color='blue',lw=3,label='observed train')
    ax2.plot(df_tst['time'],df_tst['observed'],color='green',lw=3,label='observed test')
    ax2.plot(df_all['time'],df_all['predicted'],color='red',lw=2,linestyle='--',label='predicted')
    ax2.yaxis.set_major_formatter(formatter1)
    ax2.tick_params(axis="y", colors="black", labelcolor="black",
                      labelsize=18,width=3,length=10)
    ax2.yaxis.get_offset_text().set_fontsize(18)
    ax2.set_ylabel("Discharge ($m^3s^{-1}$)", color="black",fontsize=20)
    ax2.tick_params(axis='x', which='major', length=10, width=1.5)
    ax2.tick_params(axis='x', which='minor', length=5, width=1)
    ax2.legend(loc='best',prop={'size':10})
    plt.show()
    fig2.savefig(f"{plot_folder}/train_test_plot.png", bbox_inches='tight')
    perf_rows = pd.DataFrame({
        'time': ['',''],
        'predicted': ['RMSE', 'NSE'],
        'observed': [rmse_tr, nse_tr]
        })
    df_tr = pd.concat([df_tr, pd.DataFrame([['', '', '']], columns=df_tr.columns)
                       , perf_rows], ignore_index=True)
    perf_rows = pd.DataFrame({
        'time': ['',''],
        'predicted': ['RMSE', 'NSE'],
        'observed': [rmse_tst, nse_tst]
        })
    df_tst = pd.concat([df_tst, pd.DataFrame([['', '', '']], columns=df_tst.columns)
                       , perf_rows], ignore_index=True)
    perf_rows = pd.DataFrame({
        'time': ['','','',''],
        'predicted': ['RMSE_train', 'NSE_train','RMSE_test', 'NSE_test'],
        'observed': [rmse_tr, nse_tr, rmse_tst, nse_tst]
        })
    df_all = pd.concat([df_all, pd.DataFrame([['', '', '']], columns=df_all.columns)
                       , perf_rows], ignore_index=True)
    os.makedirs(out_folder, exist_ok=True)
    df_tr.to_csv(os.path.join(out_folder,'train.csv'),index=False,)
    df_tst.to_csv(os.path.join(out_folder,'test.csv'),index=False,)
    df_all.to_csv(os.path.join(out_folder,'train_test.csv'),index=False,)
    torch.save(model.state_dict(), os.path.join(out_folder,"model_state.pth"))
    return model, df_tr,df_tst, df_all

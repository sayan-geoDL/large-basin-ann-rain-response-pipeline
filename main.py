#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 21 16:07:59 2025

@author: sayan
"""

import yaml
import argparse
import logging
import sys

# Import your 3 separate scripts as modules (assuming functions are modularized there)
from basin_response_ann_validation import preprocess_data as preprocess_cv, train_model
from basin_response_ann_trainer import trainer
from basin_response_ann_predict import preprocess_data as preprocess_pred, predict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("main.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
def run_cross_validation(config):
    logger.info("Running cross-validation...")
    ds_x = config["data"]["rain_train"]
    ds_y = config["data"]["discharge_train"]
    ds_z = config["data"]["temperature_train"]
    shp = config["data"]["shape_file"]
    train_time = config["cross_validation"]["train_time"]
    months = config["cross_validation"]["months"]
    lr = config["cross_validation"]["learning_rate"]
    epochs = config["cross_validation"]["epochs"]
    w_dec = config["cross_validation"]["weight_decay"]
    neurons = config["cross_validation"]["neuron_list"]
    a_func= config["cross_validation"]["activation_function"]
    out_folder=config["output"]["out_dir"]
    save_fig= config["cross_validation"]['save_fig']
    plot_fold=config["output"]["plots_dir"]
    x, y, norm_stats, time_index = preprocess_cv(ds_x, ds_y, ds_z, shp, train_time, months)
    train_model(x, y, neurons, lr, epochs, norm_stats, time_index, w_dec=w_dec,
                a_func=a_func,cv_folder=out_folder,sav=save_fig,plot_folder=plot_fold)

def run_full_training(config):
    logger.info("Running full training...")
    ds_x = config["data"]["rain_train"]
    ds_y = config["data"]["discharge_train"]
    ds_z = config["data"]["temperature_train"]
    shp = config["data"]["shape_file"]
    train_time = config["full_train"]["train_time"]
    test_time = config["full_train"]["test_time"]
    months = config["cross_validation"]["months"]
    lr = config["full_train"]["learning_rate"]
    epochs = config["full_train"]["epochs"]
    w_dec = config["full_train"]["weight_decay"]
    neuron_n = config["full_train"]["hidden_neuron"]
    out_folder=config["output"]["out_dir"]
    a_func= config["full_train"]["activation_function"]
    plot_fold=config["output"]["plots_dir"]
    from basin_response_ann_trainer import preprocess_data as preprocess_full
    x_tr, y_tr, x_tst, y_tst, stats, time_tr, time_tst = preprocess_full(
        ds_x, ds_y, ds_z, shp, train_time, test_time, months,out_folder=out_folder)
    
    trainer(x_tr, y_tr, x_tst, y_tst, epochs, lr, neuron_n,
            stats, time_tr, time_tst, w_dec=w_dec,a_func=a_func,plot_folder=plot_fold)

def run_prediction(config):
    logger.info("Running prediction...")
    ds_x = config["data"]["rain_predict"]
    ds_z = config["data"]["temperature_predict"]
    shp = config["data"]["shape_file"]
    pred_time = config["prediction"]["predict_time"]
    months = config["cross_validation"]["months"]
    neuron_n = config["full_train"]["hidden_neuron"]
    a_func= config["full_train"]["activation_function"]
    out_folder=config["output"]["out_dir"]
    
    x_pr, time_ind = preprocess_pred(ds_x, ds_z, shp, pred_time, months,out_folder=out_folder
                                     ,stat_path=out_folder)
    predict(x_pr, neuron_n, time_ind, a_func=a_func)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ANN hydrology model runner")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--mode", type=str, choices=["cv", "train", "predict"], required=True,
                        help="Choose which step to run: cv (cross-validation), train (retrain full), predict")

    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    try:
        if args.mode == "cv":
            run_cross_validation(config)
        elif args.mode == "train":
            run_full_training(config)
        elif args.mode == "predict":
            run_prediction(config)
    except Exception:
        logger.error("Uncaught exception occurred", exc_info=True)
        sys.exit(1)  # Optional: exit with error code 1 to indicate failure

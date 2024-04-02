from logging import Logger
import pandas as pd
import numpy as np
import torch
import joblib

from logging import Logger
import os
import sys
from typing import Callable, Dict, List, Tuple

from .run_training import run_training
from chemprop.args import TrainArgs
from chemprop.constants import TEST_SCORES_FILE_NAME, TRAIN_LOGGER_NAME
from chemprop.data import get_data, get_task_names, MoleculeDataset, validate_dataset_type
from chemprop.utils import create_logger, makedirs, timeit
from chemprop.features import set_extra_atom_fdim, set_extra_bond_fdim, set_explicit_h, set_reaction

from chemprop.train.run_training import get_xgboost_feature
from chemprop.morgan_feature import get_morgan_feature,xgboost_cv,xgb_cv_more,xgb_regre_cv,xgb_regre_more,svm_knn_rf_class,svm_knn_rf_regre,svm_knn_rf_class_more,svm_knn_rf_regre_more


def single_dhtnn_xgb(args: TrainArgs) -> Tuple[float, float]:
    """with no k-fold cross validation"""

    logger = create_logger(name=TRAIN_LOGGER_NAME, save_dir=args.save_dir, quiet=args.quiet)
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Initialize relevant variables
    init_seed = args.seed
    save_dir = args.save_dir
    args.task_names = get_task_names(path=args.data_path, smiles_columns=args.smiles_columns,
                                     target_columns=args.target_columns, ignore_columns=args.ignore_columns)

    # Print command line
    debug('Command line')
    debug(f'python {" ".join(sys.argv)}')

    # Print args
    debug('Args')
    debug(args)

    # Save args
    makedirs(args.save_dir)
    args.save(os.path.join(args.save_dir, 'args.json'))

    # set explicit H option and reaction option
    set_explicit_h(args.explicit_h)
    set_reaction(args.reaction, args.reaction_mode)

    # Get data
    debug('Loading data')
    data = get_data(
        path=args.data_path,
        args=args,
        smiles_columns=args.smiles_columns,
        logger=logger,
        skip_none_targets=True
    )
    validate_dataset_type(data, dataset_type=args.dataset_type)
    args.features_size = data.features_size()

    if args.atom_descriptors == 'descriptor':
        args.atom_descriptors_size = data.atom_descriptors_size()
        args.ffn_hidden_size += args.atom_descriptors_size
    elif args.atom_descriptors == 'feature':
        args.atom_features_size = data.atom_features_size()
        set_extra_atom_fdim(args.atom_features_size)
    if args.bond_features_path is not None:
        args.bond_features_size = data.bond_features_size()
        set_extra_bond_fdim(args.bond_features_size)

    debug(f'Number of tasks = {args.num_tasks}')

    if args.target_weights is not None and len(args.target_weights) != args.num_tasks:
        raise ValueError(
            'The number of provided target weights must match the number and order of the prediction tasks')

    # Run training
    scores_df = pd.DataFrame()
    args.seed = init_seed
    args.save_dir = os.path.join(save_dir)
    makedirs(args.save_dir)
    data.reset_features_and_targets()
    test_scores_path = os.path.join(args.save_dir, 'test_scores.json')

    dhtnn_scores,model,scaler,test_preds= run_training(args, data, logger)

    print('the ensemble_scores of dhtnn models :',dhtnn_scores)
    # if args.loss_save:
    #     # df.to_csv('/home/cxw/python_work/paper_gcn/dmpnn_epoch_loss/'+args.protein+'_loss.csv',index=None)
    #     df.to_csv(args.protein+'loss.csv',index=None)
    #     # break
    train_target, train_feature, val_target, val_feature, test_target, test_feature,train_smiles,val_smiles,test_smiles = get_xgboost_feature(args,data, logger,model)
    train_target = pd.DataFrame(train_target)
    train_feature = pd.DataFrame(train_feature)
    val_target = pd.DataFrame(val_target)
    val_feature = pd.DataFrame(val_feature)
    test_target = pd.DataFrame(test_target)
    test_feature = pd.DataFrame(test_feature)
    train_morgan_feature = get_morgan_feature(train_smiles)
    val_morgan_feature = get_morgan_feature(val_smiles)
    test_morgan_feature = get_morgan_feature(test_smiles)
    max_depth_numbers = [2,4,6,8,10]
    learning_rate_numbers = [0.01,0.05,0.1,0.15,0.2]
    min_child_weight_numbers = [2,4,6,8,10]
    if args.dataset_type == 'classification':
        if test_target.shape[1]==1:
            scores = xgboost_cv(max_depth_numbers,learning_rate_numbers,min_child_weight_numbers,
                                   train_feature, train_target,val_feature, val_target,test_feature,test_target,
                                   train_morgan_feature,val_morgan_feature,test_morgan_feature,test_preds)
        else:
            scores = xgb_cv_more(max_depth_numbers,learning_rate_numbers,min_child_weight_numbers,
                                   train_feature, train_target,val_feature, val_target,test_feature,test_target,
                                   train_morgan_feature,val_morgan_feature,test_morgan_feature,test_preds)
        scores.columns = ['type','max_depth','learning_rate','min_child_weight','auc','sn','sp','acc']
        scores_df = pd.concat([scores_df,scores])
    elif args.dataset_type == 'regression':
        if test_target.shape[1]==1:
            scores = xgb_regre_cv(max_depth_numbers, learning_rate_numbers, min_child_weight_numbers,
                                     train_feature, train_target, val_feature, val_target, test_feature, test_target,
                                     train_morgan_feature, val_morgan_feature, test_morgan_feature, test_preds, scaler)
        else:
            scores = xgb_regre_more(max_depth_numbers, learning_rate_numbers, min_child_weight_numbers,
                                     train_feature, train_target, val_feature, val_target, test_feature, test_target,
                                     train_morgan_feature, val_morgan_feature, test_morgan_feature, test_preds, scaler)
        scores.columns = ['type', 'max_depth', 'learning_rate', 'min_child_weight', 'RMSE']
        scores_df = pd.concat([scores_df,scores])

    df_groupby = scores_df.groupby(['type', 'max_depth', 'learning_rate', 'min_child_weight']).mean()
    df_groupby.to_csv(os.path.join(args.save_dir, 'scores.csv'))

    return model

def chemprop_train() -> None:
    """Parses Chemprop training arguments and trains (cross-validates) a Chemprop model.

    This is the entry point for the command line command :code:`chemprop_train`.
    """
    single_dhtnn_xgb(args=TrainArgs().parse_args())
import json
from logging import Logger
import os

from chemprop.args import TrainArgs
from chemprop.train import evaluate, evaluate_predictions

from chemprop.data import get_class_sizes, get_data,  MoleculeDataset, split_data,get_task_names,MoleculeDataLoader, set_cache_graph
from chemprop.utils import build_optimizer, build_lr_scheduler, get_loss_func, load_checkpoint, makedirs, \
    save_checkpoint, save_smiles_splits, load_frzn_model

def run_eval(args: TrainArgs,
        logger: Logger = None):

    test_data = get_data(path=args.separate_test_path,
                                 args=args,
                                 features_path=args.separate_test_features_path,
                                 atom_descriptors_path=args.separate_test_atom_descriptors_path,
                                 bond_features_path=args.separate_test_bond_features_path,
                                 smiles_columns=args.smiles_columns,
                                 logger=logger)

    args.task_names = get_task_names(path=args.separate_test_path, smiles_columns=args.smiles_columns,
                                     target_columns=args.target_columns, ignore_columns=args.ignore_columns)
    # Set up test set evaluation
    #test_smiles, test_targets = test_data.smiles(), test_data.targets()

    test_data_loader = MoleculeDataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False
    )

    model = load_checkpoint(os.path.join(args.save_dir, 'model_0\model.pt'), device=args.device, logger=logger)

    test_scores = evaluate(
        #model=model,
        data_loader=test_data_loader,
        num_tasks=args.num_tasks,
        metrics=args.metrics,
        dataset_type=args.dataset_type,
        args=args,
        logger=logger
    )

    # # Optionally save test preds
    # if args.save_preds:
    #     test_preds_dataframe = pd.DataFrame(data={'smiles': test_data.smiles()})
    #
    #     for i, task_name in enumerate(args.task_names):
    #         test_preds_dataframe[task_name] = [pred[i] for pred in test_preds]
    #
    #     test_preds_dataframe.to_csv(os.path.join(args.save_dir, 'test_preds.csv'), index=False)

    for (model_idx, scores) in test_scores:
        print(f'Model {model_idx} test auc = {scores:.6f}')
        # if args.show_individual_scores:
        #     # Individual test scores
        #     for task_name, test_score in zip(args.task_names, scores):
        #         print(f' test {task_name} {metric} = {test_score:.6f}')


def runing_eval() ->None:
    run_eval(args=TrainArgs().parse_args())
from typing import List

import torch
from tqdm import tqdm

from chemprop.data import MoleculeDataLoader, MoleculeDataset, StandardScaler
from chemprop.models import MoleculeModel


def predict(model: MoleculeModel,
            data_loader: MoleculeDataLoader,
            disable_progress_bar: bool = False,
            scaler: StandardScaler = None,
            featurizer:bool=False) -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A :class:`~chemprop.models.model.MoleculeModel`.
    :param data_loader: A :class:`~chemprop.data.data.MoleculeDataLoader`.
    :param disable_progress_bar: Whether to disable the progress bar.
    :param scaler: A :class:`~chemprop.features.scaler.StandardScaler` object fit on the training targets.
    :return: A list of lists of predictions. The outer list is molecules while the inner list is tasks.
    """
    model.eval()

    vector_list=[]
    for batch in tqdm(data_loader, disable=disable_progress_bar, leave=False):
        # Prepare batch
        batch: MoleculeDataset
        mol_batch, features_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch = \
            batch.batch_graph(), batch.features(), batch.atom_descriptors(), batch.atom_features(), batch.bond_features()

        # Make predictions or features
        with torch.no_grad():
            if not featurizer:
                vectors= model(mol_batch, features_batch, atom_descriptors_batch,
                                atom_features_batch, bond_features_batch)
            else :vectors= model.featurize(mol_batch, features_batch, atom_descriptors_batch,
                                atom_features_batch, bond_features_batch)

        vectors = vectors.cpu().numpy()
        # Inverse scale if regression
        if scaler is not None and not featurizer:
            vectors = scaler.inverse_transform(vectors)

        # Collect vectors
        vectors = vectors.tolist()
        vector_list.extend(vectors)

    return vector_list

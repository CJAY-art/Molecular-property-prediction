from rdkit import Chem
from rdkit.Chem.Draw import MolsToGridImage
import pandas as pd

data=pd.read_csv('show.csv')
ori_smiles=data['smiles'].tolist()
sub_smiles=data['rationale'].tolist()
ori_mol_list = [Chem.MolFromSmiles(x) for x in ori_smiles]
sub_mol_list = [Chem.MolFromSmiles(x) for x in sub_smiles]
plot1=MolsToGridImage(ori_mol_list)
plot1.show()
plot2=MolsToGridImage(mols=sub_mol_list,legends=sub_smiles)
plot2.show()
plot2.save('sub.png')
match_list=[]
for i in range(len(ori_mol_list)):
    match=ori_mol_list[i].GetSubstructMatch(sub_mol_list[i])
    match_list.append(match)

plot3=MolsToGridImage(mols = ori_mol_list, highlightAtomLists=match_list, molsPerRow=3,legends=ori_smiles)
plot3.show()
plot3.save('sub_ori.png')





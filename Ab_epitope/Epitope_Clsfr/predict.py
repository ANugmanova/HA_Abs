import torch
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import os
import argparse
from model import Epitope_Clsfr
from utils import get_dataset, get_dataset_from_df
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from tqdm import tqdm
# Usage
'''
python predict.py
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", default=32, type=int) 
    parser.add_argument("-lr", "--learning_rate", default=1e-4, type=float)
    parser.add_argument("-c", "--classes", default=7, type=int)
    parser.add_argument("-lm", "--language_model", default='mBLM', type=str)
    parser.add_argument("-l", "--layers", default=1, type=int)
    parser.add_argument("-hd", "--hidden_dim", default=768, type=int)
    parser.add_argument("-dp", "--dataframe_path", default='result/Flu_unknown.csv', type=str)
    parser.add_argument("-ckp", "--checkpoint_path", default='checkpoint/', type=str)
    parser.add_argument("-ckn", "--checkpoint_name", default='mBLM.ckpt', type=str)
    parser.add_argument("-n", "--name", default='mBLM_attention', type=str)
    parser.add_argument("-o", "--output_path", default='result/', type=str)

    args = parser.parse_args()

    if  args.language_model == 'mBLM':
        test_loader = get_dataset_from_df(args.output_path, args.dataframe_path, batch_size=args.batch_size, LM='mBLM')
    elif  args.language_model == 'esm2_t33_650M_UR50D':
        test_loader = get_dataset_from_df(args.output_path, args.dataframe_path, batch_size=args.batch_size, LM='esm2_t33_650M')
    elif  args.language_model == 'esm2_t6_8M_UR50D':
        test_loader = get_dataset_from_df(args.output_path, args.dataframe_path, batch_size=args.batch_size, LM='esm2_t6_8M')
    else:
        test_loader = get_dataset_from_df(args.output_path, args.dataframe_path, batch_size=args.batch_size)

    filename = args.dataframe_path.split('/')[-1].split('.')[0]
    pretrained_filename = os.path.join(args.checkpoint_path+args.name+'/', args.checkpoint_name)
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = Epitope_Clsfr.load_from_checkpoint(pretrained_filename,classes=args.classes,
                                                   hidden_dim=args.hidden_dim,layers=args.layers,class_weights=None,
                                                   lm_model_name = args.language_model)
        # model.eval()
        # test on test set
        predicted_labels_ls = []
        predicted_probabilities = []

        with torch.no_grad():
            for batch in tqdm(test_loader,desc="model prediction", leave=False):
                # get the inputs and labels
                inputs, labels, _ = batch

                outputs = model(inputs)
                # get model prediction probabilities
                probs = torch.softmax(outputs,dim=1)
                # get model predicted classes index
                predicted = torch.argmax(outputs, dim=1)

                predicted_labels_ls.append(predicted)
                predicted_probabilities.append(probs)

        predicted_all = np.concatenate(predicted_labels_ls)
        probabilities_all = np.concatenate(predicted_probabilities)

        # read df
        if args.dataframe_path.split('.')[-1].lower() == 'csv':
            df = pd.read_csv(args.dataframe_path)

        elif args.dataframe_path.split('.')[-1].lower() == 'tsv':
            df = pd.read_csv(args.dataframe_path,sep='\t')
        elif args.dataframe_path.split('.')[-1].lower() == 'xlsx':
            df = pd.read_excel(args.dataframe_path)
        else:
            print(f"Error: unsupported file type for {args.dataframe_pat}")
        # add the predicted class and probability to the DataFrame

        classes = ["HA:Head", "HA:Stem","HIV", "S:NTD", "S:RBD", "S:S2", "Others"]

        df['predicted_class'] = [classes[pred] for pred in predicted_all]
        df['predicted_probability'] = probabilities_all[np.arange(len(predicted_all)), predicted_all]
        df.to_csv(f'{args.output_path}/{filename}_prediction.tsv',sep='\t')


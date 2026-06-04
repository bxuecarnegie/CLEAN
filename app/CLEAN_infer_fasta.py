import argparse
import os
from CLEAN.utils import *
from CLEAN.infer import infer_maxsep

def eval_parse():
    # only argument passed is the fasta file name to infer
    # located in ./data/[args.fasta_data].fasta
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--fasta_data', type=str, required=True)
    # located in ./data/[args.train_data].csv
    # located in ./data/model/[args.train_data].pth
    parser.add_argument('-t', '--train_data', type=str, default='split100')
    parser.add_argument('-m', '--model_name', type=str)
    parser.add_argument('-g', '--gmm_path', type=str, defulat='./data/pretrained/gmm_ensumble.pkl')
    parser.add_argument('-p', '--pretrained', action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = eval_parse()
    train_data = args.train_data
    if model_name is None:
        model_name = train_data
    else:
        model_name = args.model_name
    test_data = 'inputs/' + args.fasta_data 
    # converting fasta to dummy csv file, will delete after inference
    # esm embedding are taken care of
    prepare_infer_fasta(test_data) 
    # inferred results is in
    # results/[args.fasta_data].csv
    infer_maxsep(train_data, test_data, report_metrics=False, pretrained=args.pretrained, model_name=model_name, gmm=args.gmm_path)
    # removing dummy csv file
    os.remove("data/"+ test_data +'.csv')
    

if __name__ == '__main__':
    main()

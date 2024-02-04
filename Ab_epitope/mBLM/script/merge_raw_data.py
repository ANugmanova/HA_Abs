import os
import pandas as pd
import argparse

def merge_oas_data(input_path, output_file):

    file_list = os.listdir(input_path)
    print("total files: ", len(file_list))

    df_list = []

    for i, f in enumerate(file_list):
        if i % 10 == 0:
            print(i)
        df = pd.read_csv(os.path.join(input_path, f), skiprows=[0], low_memory=False)
        df_list.append(df)

    df_all = pd.concat(df_list, ignore_index=True)
    df_all.to_csv(output_file, index=False)

    return df_all


if __name__ == "__main__":
    # Set up argparse
    parser = argparse.ArgumentParser(description='Add cluster to an Excel table')
    parser.add_argument('-p', '--input_OASpath', default='data/raw_data/oas',
                        type=str, help='the input csv file')
    parser.add_argument('-i', '--input_OASfile', default='data/raw_data/OAS_memory_paired.csv',
                        type=str, help='the input csv file')
    parser.add_argument('-o', '--output_file',  default='mBLM/result/memory_paired_Abs.csv',
                        type=str, help='the output csv file')

    parser.add_argument('-gb', '--genbank', default='data/raw_data/all_paired_antibodies_from_GB_v6.xlsx',
                        type=str, help='the paired antibody from genbank')
    
    # Parse the arguments
    args = parser.parse_args()

    if os.path.isfile(args.input_OASfile):
        OAS_DF = pd.read_csv(args.input_OASfile, low_memory=False)
    else:
        OAS_DF = merge_oas_data(args.input_OASpath, args.input_OASfile)

    genbank_df = pd.read_excel(args.genbank)
    # merge main table and the cluster id by the chosen cluster
    # Concatenate the dataframes based on the same columns ['B', 'C']
    concatenated = pd.concat([OAS_DF, genbank_df])
    deduplicated_df = concatenated.drop_duplicates(subset=['sequence_alignment_aa_heavy','sequence_alignment_aa_light'], keep='first')
    deduplicated_df = deduplicated_df.drop_duplicates(subset=['Name'], keep='first')
    # filter heavy chain length by 100
    filtered_df = deduplicated_df[deduplicated_df['sequence_alignment_aa_heavy'].str.len() > 100]
    filtered_df.to_csv(args.output_file,index=False)

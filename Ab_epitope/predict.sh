# put your data here
TEST_FASTA=data/test.fasta
TEST_CSV=result/epitope_test.tsv
OUTPUT_DIR=result

MODEL_PRETRAINED=checkpoint-539000
MODEL_TARGET_PATH=./
MODEL_TARGET_NAME=epoch=24_esm2-1.ckpt

mkdir $OUTPUT_DIR
mkdir $OUTPUT_DIR/esm2_t6_8M_embedding

python extract_mBLM_feature.py --model_location $MODEL_PRETRAINED --fasta_file $TEST_FASTA \
      --output_dir $OUTPUT_DIR/esm2_t6_8M_embedding
python ./Epitope_Clsfr/predict.py -n esm2_attention -lm esm2_t6_8M_UR50D -hd 320 \
      -ckn $MODEL_TARGET_NAME -ckp $MODEL_TARGET_PATH \
      --dataframe_path $TEST_CSV \
      --output_path  $OUTPUT_DIR




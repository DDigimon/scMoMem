
for SEED in  10 11 12 13 14 
    do
    CUDA_VISIBLE_DEVICES=4 python cite-seq_pretrain.py  --seed ${SEED} --dataset '2021' --alpha 0.3  --cell_num 30 --enc_layers 3 --model_dropout 0.1 --head_num 4 --enc_hid 256 --latent_mod 'memory' --lbeta 30 
    done
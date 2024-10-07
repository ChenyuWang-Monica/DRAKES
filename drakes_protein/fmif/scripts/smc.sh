alpha=0.03
for seed in 0 1 2
do
    echo "Running SMC... with alpha=$alpha... seed=$seed"
    CUDA_VISIBLE_DEVICES=0 python eval_finetune.py --decoding=smc --tds_alpha=$alpha --seed=$seed
done
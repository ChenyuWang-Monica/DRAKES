scale=1000
alpha=0.03
for seed in 0 1 2
do
    echo "Running TDS... with scale=$scale... alpha=$alpha... seed=$seed"
    CUDA_VISIBLE_DEVICES=2  python eval_finetune.py --decoding=tds --dps_scale=$scale --tds_alpha=$alpha --seed=$seed
done
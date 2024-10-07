scale=1000
for seed in 0 1 2
do
    echo "Running CG... with scale=$scale... seed=$seed"
    CUDA_VISIBLE_DEVICES=0 python eval_finetune.py --decoding=cg --dps_scale=$scale --seed=$seed
done
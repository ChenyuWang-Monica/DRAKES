for seed in 0 1 2
do
    echo "Running Pretrained... with seed=$seed"
    CUDA_VISIBLE_DEVICES=0 python eval_finetune.py --decoding=original --base_model=old --seed=$seed
done
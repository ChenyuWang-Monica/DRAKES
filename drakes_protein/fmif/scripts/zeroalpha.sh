for seed in 0 1 2
do
    echo "Running Zero Alpha... with seed=$seed"
    CUDA_VISIBLE_DEVICES=1 python eval_finetune.py --decoding=original --base_model=zero_alpha --seed=$seed
done
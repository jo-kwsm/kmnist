python src/utils/make_configs.py --model resnet18 resnet34 resnet50 --learning_rate 0.003 0.0003

files="./result/*"
for filepath in $files; do
    if [ -d $filepath ] ; then
        python src/train.py "${filepath}/config.yaml" --no_wandb
        python src/evaluate.py "${filepath}/config.yaml" validation
        python src/evaluate.py "${filepath}/config.yaml" test
    fi
done

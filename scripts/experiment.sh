python src/utils/make_configs.py --model benchmarknet easynet resnet18 resnet34 --degrees 0 10 --translate 0 0.1

files="./result/*"
for filepath in $files; do
    if [ -d $filepath ] ; then
        python src/train.py "${filepath}/config.yaml" --no_wandb
        python src/evaluate.py "${filepath}/config.yaml" validation
        python src/evaluate.py "${filepath}/config.yaml" test
    fi
done

python src/utils/make_final_result.py result/

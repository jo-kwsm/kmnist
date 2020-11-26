python src/utils/make_configs.py --model vgg11 --learning_rate 0.0001 0.0005 0.001 0.005 --size 224
python src/utils/make_configs.py --model resnet18 resnet34 --learning_rate 0.0001 0.0005 0.001 0.005

files="./result/*"
for filepath in $files; do
    if [ -d $filepath ] ; then
        python src/train.py "${filepath}/config.yaml" --no_wandb
        python src/evaluate.py "${filepath}/config.yaml" validation
        python src/evaluate.py "${filepath}/config.yaml" test
    fi
done

python src/utils/make_final_result.py result/

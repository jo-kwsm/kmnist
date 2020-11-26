files="./result/*"
for filepath in $files; do
    if [ -d $filepath ] ; then
        python src/evaluate.py "${filepath}/config.yaml" test
        python src/libs/graph.py "${filepath}/log.csv"
    fi
done

python src/utils/make_final_result.py result/

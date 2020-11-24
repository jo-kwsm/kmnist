files="./result/*"
for filepath in $files; do
    if [ -d $filepath ] ; then
        python src/evaluate.py "${filepath}/config.yaml" test
    fi
done

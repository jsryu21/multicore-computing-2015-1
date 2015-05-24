j=1
for k in {1..6}
do
    for i in {1..10}
    do
        thorq --add ../mat_mul -t $j -m 1024
    done
    j=$(($j * 2))
done

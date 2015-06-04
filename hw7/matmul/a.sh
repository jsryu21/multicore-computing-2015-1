for i in 1 2 4 8 16
do
    for j in 1 2 4
    do
        for k in {1..10}
        do
            echo $i $j
            make run s=$i n=$j
        done
    done
done

for i in {1..20}
do
    ./swaptions_pthread -ns 128 -sm 1000000
    gprof swaptions_pthread gmon.out > before_$i
done

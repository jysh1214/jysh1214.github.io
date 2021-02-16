set title "Performance Comparsion"
set xlabel "matrix size"
set ylabel "cost time(ms)"
set terminal png
set output "time.png"
set key center top
set xtics 1000, 1000, 10000

plot \
"plot.csv" using 1:2 with linespoints linewidth 2 title "CPU", \
"plot.csv" using 1:3 with linespoints linewidth 2 title "GPU", \

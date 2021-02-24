set title "Probability"
set xlabel "Gray Levels"
set ylabel "Probability"
set terminal png
set output "plot.png"
set style fill solid border
set xtics 0, 15, 255
set style histogram rowstacked
unset key

plot \
"plot.csv" using 2:xtic(20) with histogram title 'test.jpg'

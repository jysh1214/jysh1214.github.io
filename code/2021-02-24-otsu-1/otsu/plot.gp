set title "Histogram"
set xlabel "Gray Levels"
set ylabel "Number os Pixels"
set terminal png
set output "plot.png"
set style fill solid border
set xtics 0, 15, 255
set style histogram rowstacked
#unset key

set style line 2 linecolor rgb '#dd181f' linetype 1 linewidth 2
set parametric
const=103
set trange [0:30000]

plot \
"plot.csv" using 2:xtic(20) with histogram notitle, \
const,t  with lines linestyle 2 title 'Threshold: 103'

set terminal postscript eps color size 2.5,2
set output "gnuplot_perm_36_mirasol_mis.eps"

set pointsize 1.5

set xrange [0.9:110]
set yrange [0.05:32]
set logscale y
set grid ytics mytics lt 1 lc rgb "#EEEEEE"
set xlabel 'Filter Permeability'
set logscale x
set ylabel 'Mean MIS Time (seconds, log scale)'
set nokey
set xtics ('1%%' 1, '10%%' 10, '100%%' 100)
plot\
 "gnuplot_perm_36_mirasol_mis.dat" every ::1 using 1:2 title 'Python/Python KDT' lt 1 lw 7 lc rgb '#FF0000' pt 5 with linespoints,\
 "gnuplot_perm_36_mirasol_mis.dat" every ::1 using 1:7 title 'Python/SEJITS KDT' lt 1 lw 7 lc rgb '#228B22' pt 11 with linespoints,\
 "gnuplot_perm_36_mirasol_mis.dat" every ::1 using 1:12 title 'SEJITS/SEJITS KDT' lt 1 lw 7 lc rgb '#0000FF' pt 13 with linespoints,\
 "gnuplot_perm_36_mirasol_mis.dat" every ::1 using 1:17 title 'C++/C++ CombBLAS' lt 1 lw 7 lc rgb '#DAA520' pt 7 with linespoints

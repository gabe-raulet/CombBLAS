set title "BFS on Twitter Data (36 processes)"
set terminal png
set output "gnuplot_real_36.png"
set xrange [-0.5:3.5]
set yrange [0.01:32]
set logscale y
set grid ytics mytics lt 1 lc rgb "#EEEEEE"
set xlabel 'Twitter Input Graph'
set ylabel 'Mean BFS Time (seconds, log scale)'
set key right top
set xtics ('small' 0, 'medium' 1, 'large' 2, 'huge' 3)
plot\
 "gnuplot_real_36.dat" every ::1 using 1:2:3:4 title '' ps 0 lc rgb '#FF0000' with errorbars,\
 "gnuplot_real_36.dat" every ::1 using 1:2 title 'Python/Python KDT' lc rgb '#FF0000' with lines,\
 "gnuplot_real_36.dat" every ::1 using 1:5:6:7 title '' ps 0 lc rgb '#8B0000' with errorbars,\
 "gnuplot_real_36.dat" every ::1 using 1:5 title 'Python/SEJITS KDT' lc rgb '#8B0000' with lines,\
 "gnuplot_real_36.dat" every ::1 using 1:8:9:10 title '' ps 0 lc rgb '#0000FF' with errorbars,\
 "gnuplot_real_36.dat" every ::1 using 1:8 title 'SEJITS/SEJITS KDT' lc rgb '#0000FF' with lines,\
 "gnuplot_real_36.dat" every ::1 using 1:11:12:13 title '' ps 0 lc rgb '#DAA520' with errorbars,\
 "gnuplot_real_36.dat" every ::1 using 1:11 title 'C++/C++ CombBLAS' lc rgb '#DAA520' with lines

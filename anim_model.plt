set term gif animate delay 10   # Create an animated GIF with a 10 ms delay
set output "animation/prediction_evolution_discovery_best_cont.gif"      # Output file for the animation

set title "Evolution of Water Depth (m)"
set xlabel "x (m)"
set ylabel "y (m)"
set zrange [0:0.9]
set zlabel "Water Height (m)"
set grid

do for [t=0:50] {               # Adjust range (0:49) to match the number of timesteps
    set palette defined (0 "blue", 1 "cyan", 2 "green", 3 "yellow", 4 "red")
    splot "prediction_evolution_discovery_best_cont.dat" index t using 1:2:3 with pm3d title sprintf("Timestep %d", t)
    #set output "frame_".sprintf("%03d", t).".png"
    #splot "huz_evolution.dat" index t using 1:2:3 with pm3d
}
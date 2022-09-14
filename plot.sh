#!/bin/bash
   # Script to reproduce results


 # declare -a names=( "Push-V2"		"Reach-V2"		"Pick-Place-V2"		"Peg-Insert-Side-V2"		"Window-Open-V2"		"Drawer-Close-V2"		"Lever-Pull-V2"		"Handle-Pull-V2"		"Handle-Pull-Side-V2"		"Pick-Out-Of-Hole-V2"		"Plate-Slide-Side-V2"		"Plate-Slide-V2"		"Reach-Wall-V2"  "Soccer-V2"  	"Push-Wall-V2"  "Window-Close-V2"  	"Shelf-Place-V2"  "Sweep-V2"  "Cheetah-Vel" "Sparse-Point-Robot" "Reach-V2-Medium"  "Handle-Pull-V2-Medium" "Push-Wall-V2-Medium"  	"Window-Open-V2-Medium" )

 declare -a names=( "Push-V2"		"Reach-V2"		"Pick-Place-V2"		  "Cheetah-Vel" "Sparse-Point-Robot" "Drawer-Close-V2"		"Handle-Pull-Side-V2"	 	"Push-Wall-V2" 		"Window-Open-V2"	 )

 for name in "${names[@]}"
 do
 python plot_new_ml1_2.py --name ${name}
 done

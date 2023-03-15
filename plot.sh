#!/bin/bash
   # Script to reproduce results


#  declare -a names=( "Push-V2"		"Reach-V2"		"Pick-Place-V2"		"Peg-Insert-Side-V2"		"Window-Open-V2"		"Drawer-Close-V2"		"Lever-Pull-V2"		"Handle-Pull-V2"		"Handle-Pull-Side-V2"		"Pick-Out-Of-Hole-V2"		"Plate-Slide-Side-V2"		"Plate-Slide-V2"		"Reach-Wall-V2"  "Soccer-V2"  	"Push-Wall-V2"  "Window-Close-V2"  	"Shelf-Place-V2"  "Sweep-V2"  "Cheetah-Vel" "Sparse-Point-Robot" "Reach-V2-Medium"  "Handle-Pull-V2-Medium" "Push-Wall-V2-Medium"  	"Window-Open-V2-Medium" )

# declare -a names=( "Push-V2"		"Reach-V2"		"Pick-Place-V2"		  "Cheetah-Vel" "Sparse-Point-Robot" "Drawer-Close-V2"		"Handle-Pull-Side-V2"	 	"Push-Wall-V2" 		"Window-Open-V2"	 	"Soccer-V2" 		"Reach-Wall-V2"	 )  # initial

#  declare -a names=( 'Handle-Pull-V2'  'Lever-Pull-V2'  'Peg-Insert-Side-V2' 'Pick-Out-Of-Hole-V2' 'Plate-Slide-V2' 'Plate-Slide-Side-V2'	 "Reach-V2"   "Cheetah-Vel"  'Reach-Wall-V2'  'Soccer-V2'  'Shelf-Place-V2' 'Sweep-V2'  'Window-Close-V2' "Reach-V2-Medium"   	"Window-Open-V2-Medium"    "Handle-Pull-V2-Medium" "Push-Wall-V2-Medium" 	"Pick-Place-V2"	 )  # initial

declare -a names=( "Push-V2"		"Reach-V2"		"Pick-Place-V2"		"Peg-Insert-Side-V2"		"Window-Open-V2"		"Drawer-Close-V2"		"Lever-Pull-V2"		"Handle-Pull-V2"		"Handle-Pull-Side-V2"	"Pick-Out-Of-Hole-V2"		"Plate-Slide-Side-V2"		"Plate-Slide-V2"		"Reach-Wall-V2"  "Soccer-V2"  "Push-Wall-V2"  "Window-Close-V2"  	"Shelf-Place-V2"  "Sweep-V2"  "Cheetah-Vel"  "Sparse-Point-Robot" "Reach-V2-Medium"  "Handle-Pull-V2-Medium" "Push-Wall-V2-Medium"  	"Window-Open-V2-Medium"	 )

declare -a names=( "Push-V2"		"Reach-V2"		"Pick-Place-V2"		"Peg-Insert-Side-V2"		"Window-Open-V2"		"Drawer-Close-V2"		"Lever-Pull-V2"		"Handle-Pull-V2"		"Handle-Pull-Side-V2"	"Pick-Out-Of-Hole-V2"		"Plate-Slide-Side-V2"		"Plate-Slide-V2"		"Reach-Wall-V2"  "Soccer-V2"  "Push-Wall-V2"  "Window-Close-V2"  	"Shelf-Place-V2"  "Sweep-V2" )

declare -a names=("Pick-Out-Of-Hole-V2"	 )
declare -a names=("Push-V2"		"Pick-Place-V2"		"Drawer-Close-V2"		"Soccer-V2"		 "Sparse-Point-Robot" "Reach-V2"	"Sweep-V2Med"	"Peg-Insert-Side-V2Med"		 )
declare -a names=(  "Sparse-Point-Robot"	 "Cheetah-Vel" )
declare -a names=("Sweep-V2Med"	"Peg-Insert-Side-V2Med"		 )
#declare -a names=(  "Sparse-Point-RobotMed"	 )
#
#
#declare -a names=(  "Sparse-Point-Robot"	 )
# declare -a names=( "Pick-Place-V2"  )
 for name in "${names[@]}"
 do
 python plot_new_ml1_baseline.py --name ${name}
 done

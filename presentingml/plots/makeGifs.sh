files=$(ls energy* | sort -n -t y -k 2 -r); convert -delay 100 $files energyContours.gif

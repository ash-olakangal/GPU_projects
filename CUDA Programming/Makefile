main: main.cu
	nvcc -O3 -std=c++17 -arch=sm_89 main.cu -o main
pa4_testbench: launchStudentKernel.cu
	nvcc -O3 -std=c++17 -arch=sm_89 launchStudentKernel.cu -L/mnt/coe/workspace/ece/ece786-spr26/ -lpa4_testbench -o pa4_testbench 
grade: 
	/mnt/coe/workspace/ece/ece786-spr26/pa4_grade.sh

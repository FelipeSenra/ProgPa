CC = gcc
FLAGS = -pg -g3 -march=native -std=gnu++11 -fprofile-arcs -ftest-coverage  
DIRA1 = ".\Bin\a1
DIRA2 = ".\Bin\a1v2

All: a1

a1:
	g++ $(FLAGS) -c a1.cpp -o a1.o
	g++ $(FLAGS) a1.o -o $(DIRA1)\a1.exe

a1v2:
	g++ $(FLAGS) -c a1v2.cpp -o a1v2.o
	g++ $(FLAGS) a1v2.o -o $(DIRA2)\a1v2.exe
	g++ $(FLAGS)-msse4.2 -c a1v2.cpp -o a1v2.o
	g++ $(FLAGS)-msse4.2 a1v2.o -o $(DIRA2)\a1v3.exe
	g++ $(FLAGS)-msse2 -c a1v2.cpp -o a1v2.o
	g++ $(FLAGS)-msse2 a1v2.o -o $(DIRA2)\a1v4.exe
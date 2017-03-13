CC = gcc
FLAGS = -m32 -pg -g3 -std=gnu++11 -fprofile-arcs -ftest-coverage  


All: a1

a1:
	g++ $(FLAGS) -c a1.cpp -o a1.o
	g++ $(FLAGS) a1.o -o a1.exe

a1v2:
	g++ $(FLAGS) -c a1v2.cpp -o a1v2.o
	g++ $(FLAGS) a1v2.o -o a1v2.exe
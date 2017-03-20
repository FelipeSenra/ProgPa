CC = gcc
FLAGS = -S -pg -g3 -mavx -march=native -std=c++11 -fprofile-arcs -ftest-coverage  -fopenmp -I/opt/local/include/libomp -I/opt/local/include -Ofast -march=native --std=c++11 -Wall -W -pedantic
DIRA1 = ".\Bin\a1
DIRA2 = ".\Bin\a1v2

All: a1

a1:
	g++ $(FLAGS) -c a1.cpp -o a1.o
	g++ $(FLAGS) a1.o -o $(DIRA1)\a1.exe

a1v2:
	g++ $(FLAGS) -c a1v2.cpp -o a1v2.o
	g++ $(FLAGS) a1v2.o -o $(DIRA2)\a1v2.exe
	g++ $(FLAGS) -msse4.2 -c a1v2.cpp -o a1v2.o
	g++ $(FLAGS) -msse4.2 a1v2.o -o $(DIRA2)\a1v3.exe
	g++ $(FLAGS) -msse2 -c a1v2.cpp -o a1v2.o
	g++ $(FLAGS) -msse2 a1v2.o -o $(DIRA2)\a1v4.exe
	
clean:
	del *.o
	
#UNAME:=$(shell uname)

#CXX_Linux=g++
#CXX_Darwin=clang++
#CXX:=$(CXX_$(UNAME))

#CXXFLAGS_Linux=-fopenmp -Ofast -march=native --std=c++11 -Wall -W -pedantic
#CXXFLAGS_Darwin=-fopenmp -I/opt/local/include/libomp -I/opt/local/include -Ofast -march=native --std=c++11 -Wall -W -pedantic
#CXXFLAGS:=$(CXXFLAGS_$(UNAME))

#LIBS_Linux=-lgomp
#LIBS_Darwin=-fopenmp -L/opt/local/lib/libomp -lgomp
#LIBS:=$(LIBS_$(UNAME))

#OBJS:= a1.o

#all: a1

#a1: $(OBJS)
#	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

#%.o: %.cpp makefile
#	$(CXX) $(CXXFLAGS) $(INC) -o $@ -c $<

#clean:
#	\rm -f $(OBJS)
	
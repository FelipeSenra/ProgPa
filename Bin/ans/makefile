#UNAME:=$(shell uname)

CXX_Linux=g++
CXX_Darwin=clang++
CXX:=g++#$(CXX_$(UNAME))

CXXFLAGS_Linux=-fopenmp -Ofast -march=native --std=c++11 -Wall -W -pedantic
CXXFLAGS_Darwin=-fopenmp -I/opt/local/include/libomp -I/opt/local/include -Ofast -march=native --std=c++11 -Wall -W -pedantic
CXXFLAGS:=$(CXXFLAGS_Linux)#$(CXXFLAGS_$(UNAME))

LIBS_Linux=-lgomp
LIBS_Darwin=-fopenmp -L/opt/local/lib/libomp -lgomp
LIBS:=$(-lgomp)#$(LIBS_$(UNAME))

OBJS:= a1.o

all: a1

a1: $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

%.o: %.cpp makefile
	$(CXX) $(CXXFLAGS) $(INC) -o $@ -c $<

clean:
	\rm -f $(OBJS)

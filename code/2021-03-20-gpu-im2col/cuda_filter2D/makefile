.PHONY: clean dirs

SRC=src
OBJ=obj
LDOPENCV=`pkg-config --libs opencv4` `pkg-config --cflags opencv4` -lstdc++
LDCUDA=-lcuda -lcudart -lcublas -lcurand
CPP=g++ -Wall -std=c++17
NVCC=nvcc

all: dirs Filter2D

Filter2D: $(SRC)/main.cpp
	$(CPP) $^ -o $@ $(LDCUDA) $(LDOPENCV)

clean:
	rm -rf Filter2D $(OBJ)

dirs:
	mkdir -p $(SRC) $(OBJ)

stat:
	wc src/*

.PHONY: clean dirs

SRC=src
OBJ=obj
CVCONFIG=`pkg-config --libs opencv4` `pkg-config --cflags opencv4` -lstdc++ -Qunused-arguments -Wno-deprecated -Wno-unused-function
CLANG=clang++ -Wall -std=c++17

all: dirs Main

Main: $(SRC)/main.cpp
	$(CLANG) $^ -o $@ $(CVCONFIG)

dirs:
	mkdir -p $(SRC) $(OBJ)

clean:
	rm -rf Main $(OBJ)

run:
	./Main

stat:
	wc src/*

.PHONY: clean dirs test

SRC=src
OBJ=obj
TEST=test
OPENCV=`pkg-config --libs opencv4` `pkg-config --cflags opencv4` -lstdc++
CUDA=-lcuda -lcudart -lcublas -lcurand
GTEST=-lgtest -lgtest_main -lpthread
CPP=clang++ -Wall -std=c++11
NVCC=nvcc

all: dirs conv $(TEST)/TEST

conv: $(SRC)/main.cpp \
	$(OBJ)/im2col.o
	$(CPP) $^ -o $@ $(CUDA) $(OPENCV)

$(TEST)/TEST: $(TEST)/ut_test.cpp \
	$(OBJ)/im2col.o
	$(CPP) $^ -o $@ $(CUDA) $(OPENCV) $(GTEST)

$(OBJ)/im2col.o: $(SRC)/im2col.cu $(SRC)/im2col.h
	$(NVCC) -c $< -o $@ $(CUDA) $(OPENCV)

test:
	./$(TEST)/TEST

clean:
	rm -rf conv $(OBJ) $(TEST)/TEST

dirs:
	mkdir -p $(SRC) $(OBJ)

stat:
	wc src/*
# Makefile
CC = gcc
CFLAGS = -std=c99 -Wall -Ofast -mavx2 -march=native # -fopt-info-vec-optimized -fopt-info-vec-missed -fopt-info-vec-note -ftree-vectorize -ftree-vectorizer-verbose=2
LIBS = -lm -fopenmp -fopenmp-simd
SRC = main.c d2q9_bgk.c calc.c utils.c 
EXE=lbm

PARAMS_DIR=./data/params
OBSTACLES_DIR=./data/obstacles
RESULTS_DIR=./results
SCRIPTS_DIR=./scripts

.PHONY: all visual check clean

all: clean
	$(CC) $(CFLAGS) $(SRC) $(LIBS) -o $(EXE) 

visual: clean
	rm -rf $(RESULTS_DIR)/visual
	mkdir -p $(RESULTS_DIR)/visual
	$(CC) $(CFLAGS) $(SRC) $(LIBS) -o $(EXE) -DVISUAL
	./$(EXE) $(PARAMS_DIR)/visual.params $(OBSTACLES_DIR)/CS110.dat
	gnuplot $(SCRIPTS_DIR)/visual.plt

evaluate: all
	mkdir -p $(RESULTS_DIR)
	./$(EXE) $(PARAMS_DIR)/evaluate.params $(OBSTACLES_DIR)/CS110.dat 

plot:
	gnuplot $(SCRIPTS_DIR)/final_state.plt

clean:
	rm -f $(EXE)
	rm -f *.dat
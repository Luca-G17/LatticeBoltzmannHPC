# Makefile

EXE=d2q9-bgk

CC=icc

# C99, No Warnings, Max Optimisations, Tune for native, Include OpenMP, 
CFLAGS= -std=c99 -Ofast -mtune=native -fopenmp -xCORE-AVX2 # -ipo -fma # -parallel# -g  -Wall -pg -ggdb
LIBS = -lm
INPUT_SIZE=1024

FINAL_STATE_FILE=./final_state.dat
AV_VELS_FILE=./av_vels.dat
REF_FINAL_STATE_FILE=check/$(INPUT_SIZE)x$(INPUT_SIZE).final_state.dat
REF_AV_VELS_FILE=check/$(INPUT_SIZE)x$(INPUT_SIZE).av_vels.dat

all: $(EXE)

$(EXE): $(EXE).c
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@

check:
	python check/check.py --ref-av-vels-file=$(REF_AV_VELS_FILE) --ref-final-state-file=$(REF_FINAL_STATE_FILE) --av-vels-file=$(AV_VELS_FILE) --final-state-file=$(FINAL_STATE_FILE)

.PHONY: all check clean

clean:
	rm -f $(EXE)

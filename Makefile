CC = gcc
MPICC = mpicc
CFLAGS = -O3 -Wall -Wextra
LIBS = -lm

BIN_DIR = bin
SRC_DIR = src

SEQ_SRC = $(SRC_DIR)/sequential_vector.c
MPI_SRC = $(SRC_DIR)/mpi_vector.c

SEQ_BIN = $(BIN_DIR)/sequential_vector
MPI_BIN = $(BIN_DIR)/mpi_vector

.PHONY: all clean run_seq run_mpi benchmark plots

all: $(SEQ_BIN) $(MPI_BIN)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(SEQ_BIN): $(SEQ_SRC) $(SRC_DIR)/common.h | $(BIN_DIR)
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)

$(MPI_BIN): $(MPI_SRC) $(SRC_DIR)/common.h | $(BIN_DIR)
	$(MPICC) $(CFLAGS) $< -o $@ $(LIBS)

run_seq: $(SEQ_BIN)
	./$(SEQ_BIN) 10000000

run_mpi: $(MPI_BIN)
	mpirun -np 4 ./$(MPI_BIN) 10000000

benchmark: all
	bash results/run_benchmarks.sh

plots:
	python3 graphs/plot_results.py

presentation:
	python3 presentation/generate_presentation.py

clean:
	rm -rf $(BIN_DIR)

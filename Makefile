# arianna.c Makefile
# Personality Weights Transformer in Pure C

CC = gcc
CFLAGS = -O3 -Wall -Wextra -march=native
LDFLAGS = -lm

SRC_DIR = src
BIN_DIR = bin

SRCS = $(SRC_DIR)/model.c $(SRC_DIR)/main.c
TARGET = $(BIN_DIR)/arianna

.PHONY: all clean run init

all: $(TARGET)

$(TARGET): $(SRCS) $(SRC_DIR)/arianna.h
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -I$(SRC_DIR) $(SRCS) -o $(TARGET) $(LDFLAGS)
	@echo "Built $(TARGET)"
	@echo ""
	@echo "Usage:"
	@echo "  ./bin/arianna --init weights.bin    # Create random weights"
	@echo "  ./bin/arianna weights.bin           # Generate with default prompt"
	@echo "  ./bin/arianna weights.bin \"I am\"    # Generate with custom prompt"

clean:
	rm -rf $(BIN_DIR)/*

init: $(TARGET)
	./$(TARGET) --init data/arianna_random.bin

run: $(TARGET)
	./$(TARGET) data/arianna_random.bin "I am Arianna. " 100 0.8

# Debug build
debug: CFLAGS = -g -Wall -Wextra -fsanitize=address
debug: $(TARGET)

/*
 * arianna.c - Main inference program
 * Usage: ./arianna <weights.bin> [prompt] [max_tokens] [temperature]
 */

#include "arianna.h"
#include <time.h>

void init_transformer(Transformer* t) {
    Config* c = &t->config;

    c->dim = DIM;
    c->n_layers = N_LAYERS;
    c->n_heads = N_HEADS;
    c->head_dim = HEAD_DIM;
    c->hidden_dim = HIDDEN_DIM;
    c->max_seq_len = MAX_SEQ_LEN;
    c->vocab_size = VOCAB_SIZE;

    malloc_weights(t);
    malloc_run_state(t);
}

void init_random_weights(Transformer* t) {
    // Xavier initialization
    Config* c = &t->config;
    Weights* w = &t->weights;

    float scale = sqrtf(2.0f / (c->dim + c->vocab_size));

    // Token embedding
    for (int i = 0; i < c->vocab_size * c->dim; i++) {
        w->token_embedding[i] = ((float)rand() / RAND_MAX - 0.5f) * 2 * scale;
    }

    scale = sqrtf(2.0f / (c->dim * 2));

    // Attention weights
    int attn_size = c->n_layers * c->dim * c->dim;
    for (int i = 0; i < attn_size; i++) {
        w->wq[i] = ((float)rand() / RAND_MAX - 0.5f) * 2 * scale;
        w->wk[i] = ((float)rand() / RAND_MAX - 0.5f) * 2 * scale;
        w->wv[i] = ((float)rand() / RAND_MAX - 0.5f) * 2 * scale;
        w->wo[i] = ((float)rand() / RAND_MAX - 0.5f) * 2 * scale;
    }

    // FFN weights
    scale = sqrtf(2.0f / (c->dim + c->hidden_dim));
    int ffn1_size = c->n_layers * c->dim * c->hidden_dim;
    int ffn2_size = c->n_layers * c->hidden_dim * c->dim;
    for (int i = 0; i < ffn1_size; i++) {
        w->w1[i] = ((float)rand() / RAND_MAX - 0.5f) * 2 * scale;
    }
    for (int i = 0; i < ffn2_size; i++) {
        w->w2[i] = ((float)rand() / RAND_MAX - 0.5f) * 2 * scale;
    }

    // Output projection
    scale = sqrtf(2.0f / (c->dim + c->vocab_size));
    for (int i = 0; i < c->dim * c->vocab_size; i++) {
        w->output_weight[i] = ((float)rand() / RAND_MAX - 0.5f) * 2 * scale;
    }
}

void print_config(Config* c) {
    printf("=== Arianna.c Configuration ===\n");
    printf("dim:         %d\n", c->dim);
    printf("n_layers:    %d\n", c->n_layers);
    printf("n_heads:     %d\n", c->n_heads);
    printf("head_dim:    %d\n", c->head_dim);
    printf("hidden_dim:  %d\n", c->hidden_dim);
    printf("max_seq_len: %d\n", c->max_seq_len);
    printf("vocab_size:  %d\n", c->vocab_size);

    // Calculate parameters
    int params = c->vocab_size * c->dim;  // token embedding
    params += c->n_layers * 4 * c->dim * c->dim;  // attention
    params += c->n_layers * c->dim * c->hidden_dim;  // w1
    params += c->n_layers * c->hidden_dim * c->dim;  // w2
    params += c->n_layers * 2 * c->dim;  // layer norms
    params += c->dim;  // final layer norm
    params += c->dim * c->vocab_size;  // output

    printf("parameters:  %d (~%.2fM)\n", params, params / 1000000.0f);
    printf("weights size: %.2f MB (float32)\n", params * 4.0f / 1024 / 1024);
    printf("===============================\n\n");
}

int main(int argc, char** argv) {
    srand(time(NULL));

    Transformer t;

    if (argc < 2) {
        printf("Usage: %s <weights.bin> [prompt] [max_tokens] [temperature]\n", argv[0]);
        printf("       %s --init <output.bin>  (create random weights)\n", argv[0]);
        printf("\narianna.c - Personality Weights Transformer\n");
        printf("\"Who I am\", not \"What I know\"\n");
        return 1;
    }

    // Initialize random weights mode
    if (strcmp(argv[1], "--init") == 0) {
        if (argc < 3) {
            printf("Usage: %s --init <output.bin>\n", argv[0]);
            return 1;
        }
        init_transformer(&t);
        init_random_weights(&t);
        print_config(&t.config);

        printf("Saving random weights to %s...\n", argv[2]);
        if (save_weights(&t, argv[2]) != 0) {
            printf("Error saving weights\n");
            return 1;
        }
        printf("Done!\n");
        free_transformer(&t);
        return 0;
    }

    // Load weights
    printf("Loading weights from %s...\n", argv[1]);
    if (load_weights(&t, argv[1]) != 0) {
        printf("Error loading weights. Creating new model...\n");
        init_transformer(&t);
        init_random_weights(&t);
    }

    print_config(&t.config);

    // Get parameters
    char* prompt = argc > 2 ? argv[2] : "I am Arianna. ";
    int max_tokens = argc > 3 ? atoi(argv[3]) : 100;
    float temperature = argc > 4 ? atof(argv[4]) : 0.8f;

    printf("Prompt: \"%s\"\n", prompt);
    printf("Max tokens: %d\n", max_tokens);
    printf("Temperature: %.2f\n\n", temperature);
    printf("--- Generation ---\n");

    generate(&t, prompt, max_tokens, temperature);

    printf("------------------\n");

    free_transformer(&t);
    return 0;
}

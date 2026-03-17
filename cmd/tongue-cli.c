/*
 * tongue-cli.c — CLI wrapper for Arianna tongue inference (libarianna.so)
 *
 * Usage: ./tongue-cli <weights.gguf> <prompt> [max_tokens] [temp]
 *
 * Build:
 *   cd golib && go build -buildmode=c-shared -o ../lib/libarianna.so .
 *   cc -o tongue-cli cmd/tongue-cli.c -Llib -larianna -Wl,-rpath,'$ORIGIN/lib'
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Go c-shared exports */
extern int tongue_init(char* weightsPath);
extern void tongue_free(void);
extern int tongue_generate(char* prompt, char* outputBuf, int maxOutputLen,
                           int maxTokens, float temperature, float topP,
                           char* anchorPrompt);

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <weights.gguf> <prompt> [max_tokens] [temp]\n", argv[0]);
        return 1;
    }

    const char* weights = argv[1];
    const char* prompt = argv[2];
    int max_tokens = argc > 3 ? atoi(argv[3]) : 256;
    float temp = argc > 4 ? (float)atof(argv[4]) : 0.85f;

    int rc = tongue_init((char*)weights);
    if (rc != 0) {
        fprintf(stderr, "[tongue-cli] init failed (rc=%d)\n", rc);
        return 1;
    }

    char buf[16384];
    memset(buf, 0, sizeof(buf));
    int n = tongue_generate((char*)prompt, buf, sizeof(buf), max_tokens, temp, 0.9f, NULL);

    if (n > 0) {
        printf("%s\n", buf);
    }

    tongue_free();
    return 0;
}

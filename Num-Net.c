// Reduced console flooding: using carriage return to overwrite line
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define INPUT_SIZE 4
#define HIDDEN_SIZE 24
#define OUTPUT_SIZE 10
#define POPULATION_SIZE 50
#define TOURNAMENT_SIZE 6
#define MUTATION_RATE 0.1f
#define MAX_TARGET 9
#define STALL_LIMIT 500

float learning_decay = 0.2f;

typedef struct {
    float input_hidden[INPUT_SIZE][HIDDEN_SIZE];
    float hidden_output[HIDDEN_SIZE][OUTPUT_SIZE];
} Network;

typedef struct {
    Network net;
    float fitness;
} Individual;

void int_to_binary(int n, int* out) {
    for (int i = 0; i < INPUT_SIZE; ++i)
        out[INPUT_SIZE - 1 - i] = (n >> i) & 1;
}

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float random_weight() {
    return ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
}

void forward(Network* net, int* input, float* output) {
    float hidden[HIDDEN_SIZE] = {0};

    for (int h = 0; h < HIDDEN_SIZE; ++h) {
        for (int i = 0; i < INPUT_SIZE; ++i)
            hidden[h] += input[i] * net->input_hidden[i][h];
        hidden[h] = sigmoid(hidden[h]);
    }

    for (int o = 0; o < OUTPUT_SIZE; ++o) {
        output[o] = 0;
        for (int h = 0; h < HIDDEN_SIZE; ++h)
            output[o] += hidden[h] * net->hidden_output[h][o];
        output[o] = sigmoid(output[o]);
    }
}

float evaluate(Network* net, int max_class) {
    float fitness = 0.0f;
    float output[OUTPUT_SIZE];
    int input[INPUT_SIZE];

    for (int i = 0; i <= max_class; ++i) {
        int_to_binary(i, input);
        forward(net, input, output);

        int max_index = 0;
        for (int j = 1; j <= max_class; ++j)
            if (output[j] > output[max_index])
                max_index = j;

        if (max_index == i)
            fitness += 2.0f;
        else
            fitness += 1.0f - fabsf((float)i - max_index) / (max_class + 1);

        fitness += output[i];
    }

    return fitness;
}

int is_perfect(Network* net, int max_class) {
    float output[OUTPUT_SIZE];
    int input[INPUT_SIZE];

    for (int i = 0; i <= max_class; ++i) {
        int_to_binary(i, input);
        forward(net, input, output);

        int max_index = 0;
        for (int j = 1; j <= max_class; ++j)
            if (output[j] > output[max_index])
                max_index = j;

        if (max_index != i)
            return 0;
    }
    return 1;
}

void mutate(Network* net, float rate) {
    for (int i = 0; i < INPUT_SIZE; ++i)
        for (int h = 0; h < HIDDEN_SIZE; ++h)
            if ((float)rand() / RAND_MAX < MUTATION_RATE)
                net->input_hidden[i][h] += random_weight() * rate;

    for (int h = 0; h < HIDDEN_SIZE; ++h)
        for (int o = 0; o < OUTPUT_SIZE; ++o)
            if ((float)rand() / RAND_MAX < MUTATION_RATE)
                net->hidden_output[h][o] += random_weight() * rate;
}

void crossover(Network* child, Network* parent1, Network* parent2) {
    for (int i = 0; i < INPUT_SIZE; ++i)
        for (int h = 0; h < HIDDEN_SIZE; ++h)
            child->input_hidden[i][h] = (rand() % 2) ? parent1->input_hidden[i][h] : parent2->input_hidden[i][h];

    for (int h = 0; h < HIDDEN_SIZE; ++h)
        for (int o = 0; o < OUTPUT_SIZE; ++o)
            child->hidden_output[h][o] = (rand() % 2) ? parent1->hidden_output[h][o] : parent2->hidden_output[h][o];
}

void copy_network(Network* dest, Network* src) {
    for (int i = 0; i < INPUT_SIZE; ++i)
        for (int h = 0; h < HIDDEN_SIZE; ++h)
            dest->input_hidden[i][h] = src->input_hidden[i][h];

    for (int h = 0; h < HIDDEN_SIZE; ++h)
        for (int o = 0; o < OUTPUT_SIZE; ++o)
            dest->hidden_output[h][o] = src->hidden_output[h][o];
}

void reset_half_population(Individual* pop) {
    for (int i = POPULATION_SIZE / 2; i < POPULATION_SIZE; ++i) {
        for (int x = 0; x < INPUT_SIZE; ++x)
            for (int y = 0; y < HIDDEN_SIZE; ++y)
                pop[i].net.input_hidden[x][y] = random_weight();
        for (int x = 0; x < HIDDEN_SIZE; ++x)
            for (int y = 0; y < OUTPUT_SIZE; ++y)
                pop[i].net.hidden_output[x][y] = random_weight();
    }
}

void print_progress_bar(int current, int total) {
    int width = 30;
    int progress = (current * width) / total;
    int percent = (current * 100) / total;

    const char* color;
    if (percent < 33) color = "\033[0;31m";
    else if (percent < 66) color = "\033[0;33m";
    else color = "\033[0;32m";

    printf(" [");
    printf("%s", color);
    for (int i = 0; i < progress; ++i) printf("=");
    printf("\033[0m");
    for (int i = progress; i < width; ++i) printf(" ");
    printf("] %3d%%", percent);
}

void log_generation(const char* logline) {
    FILE* f = fopen("generation_log.txt", "a");
    if (f) {
        fprintf(f, "%s\n", logline);
        fclose(f);
    }
}

int main() {
    printf("\033[0;34m===============================\n\033[0;36mNumber Learning Neural Network:\n\033[0;34m===============================\n\e[0;37m");
    srand((unsigned int)time(NULL));

    Individual population[POPULATION_SIZE];
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        for (int x = 0; x < INPUT_SIZE; ++x)
            for (int y = 0; y < HIDDEN_SIZE; ++y)
                population[i].net.input_hidden[x][y] = random_weight();
        for (int x = 0; x < HIDDEN_SIZE; ++x)
            for (int y = 0; y < OUTPUT_SIZE; ++y)
                population[i].net.hidden_output[x][y] = random_weight();
    }

    int current_max = 1;
    int gen = 0, stall_count = 0;
    float best_fitness_last = 0;
    time_t start_time = time(NULL);

    while (current_max <= MAX_TARGET) {
        for (int i = 0; i < POPULATION_SIZE; ++i)
            population[i].fitness = evaluate(&population[i].net, current_max);

        for (int i = 0; i < POPULATION_SIZE - 1; ++i)
            for (int j = i + 1; j < POPULATION_SIZE; ++j)
                if (population[j].fitness > population[i].fitness) {
                    Individual tmp = population[i];
                    population[i] = population[j];
                    population[j] = tmp;
                }

        Individual* best = &population[0];

        char outline[512] = {0};
        char predictions[32] = {0};

        for (int i = 0; i <= current_max; ++i) {
            int input[INPUT_SIZE];
            float output[OUTPUT_SIZE];
            int_to_binary(i, input);
            forward(&best->net, input, output);

            int max_index = 0;
            for (int j = 1; j <= current_max; ++j)
                if (output[j] > output[max_index])
                    max_index = j;

            char buf[4];
            sprintf(buf, "%d", max_index);
            strcat(predictions, buf);
        }

        sprintf(outline, "\rRange 0-%d | Generation %4d | Output: %s | Fitness: %.3f", current_max, gen, predictions, best->fitness);
        printf("%s", outline);
        print_progress_bar(current_max, MAX_TARGET);
        fflush(stdout);
        log_generation(outline);

        if (is_perfect(&best->net, current_max)) {
            time_t end_time = time(NULL);
            double duration = difftime(end_time, start_time);
            printf("\n Learned range 0 - %d in %d generations. Time: %.2fs\n", current_max, gen, duration);
            current_max++;
            gen = 0;
            stall_count = 0;
            best_fitness_last = 0;
            learning_decay *= 0.95f;
            start_time = time(NULL);
            continue;
        }

        if (best->fitness <= best_fitness_last + 0.001f)
            stall_count++;
        else {
            stall_count = 0;
            best_fitness_last = best->fitness;
        }

        if (stall_count > STALL_LIMIT) {
            printf("\n  Stalled on 0 - %d. Resetting lower half of population.\n", current_max);
            reset_half_population(population);
            stall_count = 0;
            learning_decay *= 1.1f;
        }

        Individual new_pop[POPULATION_SIZE];
        copy_network(&new_pop[0].net, &population[0].net);

        for (int i = 1; i < POPULATION_SIZE; ++i) {
            int p1 = rand() % TOURNAMENT_SIZE;
            int p2 = rand() % TOURNAMENT_SIZE;
            crossover(&new_pop[i].net, &population[p1].net, &population[p2].net);
            mutate(&new_pop[i].net, learning_decay);
        }

        for (int i = 0; i < POPULATION_SIZE; ++i)
            population[i] = new_pop[i];

        gen++;
    }

    printf("\n Successfully Learned to Count 0 Through 9!\n");
    printf("Press Enter to exit...");
    fflush(stdout);
    while (getchar() != '\n');
    getchar();

    return 0;
}

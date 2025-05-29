#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <signal.h>
#include <stdbool.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#define POPULATION_SIZE 20      // Increased population size
#define INPUT_SIZE 26          // Changed to 26 for the alphabet (a to z)
#define MUTATION_RATE 0.05      // Adjusted mutation rate
#define TOURNAMENT_SIZE 5      // Increased tournament size
#define ELITE_COUNT 2           // Number of elite individuals to preserve

typedef struct {
    char genes[INPUT_SIZE];
    int fitness;
} Individual;

void generateRandomGenes(char genes[INPUT_SIZE]) {
    for (int i = 0; i < INPUT_SIZE; ++i) {
        genes[i] = rand() % 2 ? '0' : '1';
    }
}

int calculateFitness(char genes[INPUT_SIZE]) {
    int fitness = 0;
    for (int i = 0; i < INPUT_SIZE; ++i) {
        fitness += (genes[i] == '1') ? i : 0;
    }
    return fitness;
}

void mutate(char genes[INPUT_SIZE]) {
    for (int i = 0; i < INPUT_SIZE; ++i) {
        if ((rand() / (double)RAND_MAX) < MUTATION_RATE) {
            genes[i] = (genes[i] == '0') ? '1' : '0';
        }
    }
}

void crossover(Individual parent1, Individual parent2, Individual *child) {
    int crossoverPoint = rand() % INPUT_SIZE;
    for (int i = 0; i < crossoverPoint; ++i) {
        child->genes[i] = parent1.genes[i];
    }
    for (int i = crossoverPoint; i < INPUT_SIZE; ++i) {
        child->genes[i] = parent2.genes[i];
    }
}

void printTime(double elapsedSeconds) {
    int hours = (int)(elapsedSeconds / 3600);
    int minutes = ((int)elapsedSeconds % 3600) / 60;
    int seconds = ((int)elapsedSeconds % 3600) % 60;

    printf("%02d:%02d:%02d", hours, minutes, seconds);
}

int compareIndividuals(const void *a, const void *b) {
    return ((Individual *)b)->fitness - ((Individual *)a)->fitness;
}

void tournamentSelection(Individual population[POPULATION_SIZE], Individual tournament[TOURNAMENT_SIZE]) {
    for (int i = 0; i < TOURNAMENT_SIZE; ++i) {
        int randomIndex = rand() % POPULATION_SIZE;
        tournament[i] = population[randomIndex];
    }
}

bool checkCorrectOrder(Individual individual) {
    for (int i = 0; i < INPUT_SIZE; ++i) {
        if (individual.genes[i] != '1' || individual.fitness != i) {
            return false;
        }
    }
    return true;
}

void writeLogFile(Individual population[POPULATION_SIZE], int generation, double elapsedSeconds) {
    FILE *file = fopen("logAlphaNet.txt", "w");
    if (file == NULL) {
        perror("Error opening log file");
        exit(EXIT_FAILURE);
    }

    fprintf(file, "Generation: %d\n", generation);
    fprintf(file, "Elapsed Time: %f seconds\n", elapsedSeconds);

    fprintf(file, "Best Individual:\n");
    fprintf(file, "Genes: ");
    for (int i = 0; i < INPUT_SIZE; ++i) {
        fprintf(file, "%c", population[0].genes[i] == '1' ? 'a' + i : '-');
    }
    fprintf(file, "\nFitness: %d\n", population[0].fitness);

    fclose(file);
}

void handleInterrupt(int signal) {
    // Handle interrupt logic if needed
}

int interrupted = 0;

void evolvePopulation(Individual population[POPULATION_SIZE], int *generation, clock_t *start, int targetFitness, int *learningFailed) {
    qsort(population, POPULATION_SIZE, sizeof(Individual), compareIndividuals);

#ifdef _WIN32
    system("cls");
#else
    system("clear");
#endif

    printf("\033[3A");
    printf("\033[0;34m===================================================\n\033[0;36mAlphabet Learning Neural Network:\n\033[0;34m===================================================\n\n\r\033[K\033[1;33mGeneration:\033[1;37m %d ", *generation);

    printf("\n\033[1;33mProgress: \033[1;37m");
    for (int i = 0; i < INPUT_SIZE; ++i) {
        printf("%c", population[0].genes[i] == '1' ? 'a' + i : '-');
    }

    clock_t now = clock();
    double elapsedSeconds = (double)(now - *start) / CLOCKS_PER_SEC;

    printf("\n\033[1;35mTime: \033[1;37m");
    printTime(elapsedSeconds);
    fflush(stdout);

    // Preserve elite individuals
    for (int i = 0; i < ELITE_COUNT; ++i) {
        memcpy(population[i + ELITE_COUNT].genes, population[i].genes, INPUT_SIZE);
        population[i + ELITE_COUNT].fitness = calculateFitness(population[i + ELITE_COUNT].genes);
    }

    // Generate new individuals using crossover and mutation
    for (int i = ELITE_COUNT; i < POPULATION_SIZE; ++i) {
        crossover(population[rand() % ELITE_COUNT], population[rand() % ELITE_COUNT], &population[i]);
        mutate(population[i].genes);
        population[i].fitness = calculateFitness(population[i].genes);
    }

    (*generation)++;

    // Check if the correct order is reached
    if (checkCorrectOrder(population[0])) {
        printf("\n\033[1;32mSuccessfully Learned Alphabet at generation: %d\n", *generation - 1);
        writeLogFile(population, *generation - 1, elapsedSeconds);
        // Exit here if the correct order is reached
        exit(EXIT_SUCCESS);
    }
}

int main() {
    system("title Alphabet Learning Neural Network");
    signal(SIGINT, handleInterrupt);

    srand(time(NULL));

    Individual population[POPULATION_SIZE];

    for (int i = 0; i < POPULATION_SIZE; ++i) {
        generateRandomGenes(population[i].genes);
        population[i].fitness = calculateFitness(population[i].genes);
    }

    int targetFitness = INPUT_SIZE;
    int generation = 1;
    int learningFailed = 0; // Flag to indicate learning failure

    clock_t start = clock();

    while (!interrupted && !learningFailed) { // Continue loop if not interrupted and learning has not failed
        Individual tournament[TOURNAMENT_SIZE];

        for (int i = 0; i < POPULATION_SIZE; ++i) {
            tournamentSelection(population, tournament);
            crossover(tournament[0], tournament[1], &population[i]);
            mutate(population[i].genes);
        }

        evolvePopulation(population, &generation, &start, targetFitness, &learningFailed);

#ifdef _WIN32
        Sleep(1);
#else
        usleep(500000);
#endif
    }

    if (interrupted) {
        #ifdef _WIN32
            system("cls");
        #else
            system("clear");
        #endif
        printf("\033[0;31m\rOperation Has Been Aborted!");
        printf("\033[1;37m");
        exit(EXIT_FAILURE);
    }

    printf("\n");

    return 0;
}

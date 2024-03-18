/**
 * Runs a simulation of the n-body problem in 3D.
 * 
 * To compile the program:
 *   gcc -Wall -fopenmp -O3 -march=native nbody-p.c matrix.c util.c -o nbody-p -lm
 * 
 * To run the program:
 *   ./nbody-p time-step total-time outputs-per-body input.npy output.npy [opt: num-threads]
 * where:
 *   - time-step is the amount of time between steps (Δt, in seconds)
 *   - total-time is the total amount of time to simulate (in seconds)
 *   - outputs-per-body is the number of positions to output per body
 *   - input.npy is the file describing the initial state of the system (below)
 *   - output.npy is the output of the program (see below)
 *   - last argument is an optional number of threads (a reasonable default is
 *     chosen if not provided)
 * 
 * input.npy has a n-by-7 matrix with one row per body and the columns:
 *   - mass (in kg)
 *   - initial x, y, z position (in m)
 *   - initial x, y, z velocity (in m/s)
 * 
 * output.npy is generated and has a (outputs-per-body)-by-(3n) matrix with each
 * row containing the x, y, and z positions of each of the n bodies after a
 * given timestep.
 * 
 * See the PDF for implementation details and other requirements.
 * 
 * AUTHORS:
 */

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#include "matrix.h"
#include "util.h"

#define G 6.6743015e-11
#define SOFTENING 1e-9

typedef struct {
    double mass;
    double x, y, z;
    double vx, vy, vz;
} Body;

typedef struct {
    double x;
    double y;
    double z;
} Point;

double distance(Point p1, Point p2) {
    return sqrt(((p2.x - p1.x) * (p2.x - p1.x)) + ((p2.y - p1.y)* (p2.y - p1.y)) + ((p2.z - p1.z)*(p2.z - p1.z))) + SOFTENING;
}

double calculateGravitationalForce(double mass1, double mass2, double distance) {
    return G * ((mass1 * mass2) / pow(distance, 2));
}

static inline Point calculateAcceleration(Point netForce, double mass) {
    Point acceleration = {netForce.x / mass, netForce.y / mass, netForce.z / mass};
    return acceleration;
}

void matrix_destroy(Matrix* matrix) {
    if (matrix != NULL) {
        free(matrix->data);
        free(matrix);
    }
}

int main(int argc, const char* argv[]) {
    if (argc != 6 && argc != 7) {
        fprintf(stderr, "usage: %s time-step total-time outputs-per-body input.npy output.npy [num-threads]\n", argv[0]);
        return 1;
    }

    double time_step = atof(argv[1]), total_time = atof(argv[2]);
    if (time_step <= 0 || total_time <= 0 || time_step > total_time) {
        fprintf(stderr, "time-step and total-time must be positive with total-time > time-step\n");
        return 1;
    }

    size_t num_outputs = atoi(argv[3]);
    if (num_outputs <= 0) {
        fprintf(stderr, "outputs-per-body must be positive\n");
        return 1;
    }

    Matrix* input = matrix_from_npy_path(argv[4]);
    if (input == NULL) {
        perror("error reading input");
        return 1;
    }

    if (input->cols != 7) {
        fprintf(stderr, "input.npy must have 7 columns\n");
        matrix_destroy(input);
        return 1;
    }

    size_t n = input->rows;
    if (n == 0) {
        fprintf(stderr, "input.npy must have at least 1 row\n");
        matrix_destroy(input);
        return 1;
    }

    size_t num_steps = (size_t)(total_time / time_step + 0.5);
    if (num_steps < num_outputs) {
        num_outputs = 1;
    }

    size_t output_steps = num_steps / num_outputs;
    num_outputs = (num_steps + output_steps - 1) / output_steps;

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    Matrix* output = matrix_create_raw(num_outputs, 3 * n);
    if (output == NULL) {
        perror("error creating output matrix");
        matrix_destroy(input);
        return 1;
    }

    matrix_fill_zeros(output);

    for (size_t i = 0; i < n; i++) {
        output->data[i * 3] = input->data[2 + (i * 6)];
        output->data[i * 3 + 1] = input->data[3 + (i * 6)];
        output->data[i * 3 + 2] = input->data[4 + (i * 6)];
    }

    Body* bodies = (Body*)malloc(n * sizeof(Body));
    if (bodies == NULL) {
        perror("error allocating memory for bodies");
        matrix_destroy(input);
        matrix_destroy(output);
        return 1;
    }

    for (size_t i = 0; i < n; i++) {
        bodies[i].mass = input->data[i * 7];
        bodies[i].x = input->data[i * 7 + 1];
        bodies[i].y = input->data[i * 7 + 2];
        bodies[i].z = input->data[i * 7 + 3];
        bodies[i].vx = input->data[i * 7 + 4];
        bodies[i].vy = input->data[i * 7 + 5];
        bodies[i].vz = input->data[i * 7 + 6];
    }

    #pragma omp parallel for
    for (size_t t = 1; t < num_steps; t++) {
        for (size_t i = 0; i < n; i++) {
            Point netForce = {0.0, 0.0, 0.0};
            Point particle_i = {bodies[i].x, bodies[i].y, bodies[i].z};

            for (size_t j = 0; j < n; j++) {
                if (i == j) continue;
                Point particle_j = {bodies[j].x, bodies[j].y, bodies[j].z};
                double dist = distance(particle_i, particle_j);
                double force = calculateGravitationalForce(bodies[i].mass, bodies[j].mass, dist);

                double forceX = force * (particle_j.x - particle_i.x) / dist;
                double forceY = force * (particle_j.y - particle_i.y) / dist;
                double forceZ = force * (particle_j.z - particle_i.z) / dist;

                netForce.x += forceX;
                netForce.y += forceY;
                netForce.z += forceZ;
            }

            Point acceleration = calculateAcceleration(netForce, bodies[i].mass);
            bodies[i].vx += acceleration.x * time_step;
            bodies[i].vy += acceleration.y * time_step;
            bodies[i].vz += acceleration.z * time_step;

            bodies[i].x += bodies[i].vx * time_step;
            bodies[i].y += bodies[i].vy * time_step;
            bodies[i].z += bodies[i].vz * time_step;
        }

        if (t % output_steps == 0) {
            for (size_t j = 0; j < n; j++) {
                output->data[(t / output_steps) * 3 * n + j * 3] = bodies[j].x;
                output->data[(t / output_steps) * 3 * n + j * 3 + 1] = bodies[j].y;
                output->data[(t / output_steps) * 3 * n + j * 3 + 2] = bodies[j].z;
            }
        }
    }

    if (num_steps % output_steps != 0) {
        for (size_t j = 0; j < n; j++) {
            output->data[(num_outputs - 1) * 3 * n + j * 3] = bodies[j].x;
            output->data[(num_outputs - 1) * 3 * n + j * 3 + 1] = bodies[j].y;
            output->data[(num_outputs - 1) * 3 * n + j * 3 + 2] = bodies[j].z;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed_time = get_time_diff(&start, &end);
    printf("%f secs\n", elapsed_time);

    matrix_to_npy_path(argv[5], output);

    free(bodies);
    matrix_destroy(input);
    matrix_destroy(output);

    return 0;
}

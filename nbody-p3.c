/**
 * Runs a simulation of the n-body problem in 3D.
 * 
 * To compile the program:
 *   gcc -Wall -O3 -march=native nbody-s.c matrix.c util.c -o nbody-s -lm
 * 
 * To run the program:
 *   ./nbody-s time-step total-time outputs-per-body input.npy output.npy
 * where:
 *   - time-step is the amount of time between steps (Δt, in seconds)
 *   - total-time is the total amount of time to simulate (in seconds)
 *   - outputs-per-body is the number of positions to output per body
 *   - input.npy is the file describing the initial state of the system (below)
 *   - output.npy is the output of the program (see below)
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
 * 
 * TODO: Cleanup
 * Check times
 * 
 * 
 * 
 * 
 * 
 * AUTHORS:
 */

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#include "matrix.h"
//#include "matrix.c"
#include "util.h"

// Gravitational Constant in N m^2 / kg^2 or m^3 / kg / s^2
#define G 6.6743015e-11

// Softening factor to reduce divide-by-near-zero effects
#define SOFTENING 1e-9

typedef struct body {
    double mass;
    double x, y, z;
    double vx, vy, vz;
} body;

typedef struct {
    double x;
    double y;
    double z;
} Point;


// Function to calculate the distance between two points
double distance(Point p1, Point p2) {
    return sqrt(((p2.x-p1.x)*(p2.x-p1.x)) + ((p2.y-p1.y)*(p2.y-p1.y)) + ((p2.z-p1.z)*(p2.z-p1.z)) + SOFTENING); 
}

// Function to calculate the gravitational force between two objects
//Newtons law of gravitation
double calculateGravitationalForce(double mass1, double mass2, double distance) {
    return G * ((mass1 * mass2) / (distance * distance));
}

// Function to calculate the net gravitational force on a particle due to several individual forces
//Superposition principle
//uses newtons third law

void calculateForcesMatrix(double masses[], double x[], double y[], double z[], double forces[], int n)
{
    Point particle_i, particle_j;
    double dist, force;
    for(int i = 0; i < n; i++)
    {
        particle_i.x = x[i];
        particle_i.y = y[i];
        particle_i.z = z[i];

        for(int j = 0; j < i; j++) 
        {
            particle_j.x = x[j];
            particle_j.y = y[j];
            particle_j.z = z[j];
            dist = distance(particle_i, particle_j);
            force = calculateGravitationalForce(masses[i], masses[j], dist);
            
            forces[i*3+0] += force * (particle_j.x - particle_i.x) / dist;
            forces[i*3+1] += force * (particle_j.y - particle_i.y) / dist;
            forces[i*3+2] += force * (particle_j.z - particle_i.z) / dist;
            forces[j*3+0] -= force * (particle_j.x - particle_i.x) / dist;
            forces[j*3+1] -= force * (particle_j.y - particle_i.y) / dist;
            forces[j*3+2] -= force * (particle_j.z - particle_i.z) / dist;
        }
    }    
}



//using the force matrix, calculate the net forces acting on each body
Point calculateNetForce(double force, Point particle_i, Point particle_j)
{
    double dist = distance(particle_i, particle_j);
    
    Point netForce = {0.0, 0.0, 0.0};
    netForce.x = force * (particle_j.x - particle_i.x) / dist;
    netForce.y = force * (particle_j.y - particle_i.y) / dist;
    netForce.z = force * (particle_j.z - particle_i.z) / dist;
    return netForce;
}

//Calculates the acceleration based on the net force and mass using newtons second law of motion
static inline Point calculateAcceleration(Point netForce, double mass) {
    Point acceleration = {netForce.x / mass, netForce.y / mass, netForce.z / mass};
    return acceleration;
}

int main(int argc, const char* argv[]) {
    // parse arguments
    if (argc != 6 && argc != 7) { fprintf(stderr, "usage: %s time-step total-time outputs-per-body input.npy output.npy [num-threads]\n", argv[0]); return 1; }
    double time_step = atof(argv[1]), total_time = atof(argv[2]);
    if (time_step <= 0 || total_time <= 0 || time_step > total_time) { fprintf(stderr, "time-step and total-time must be positive with total-time > time-step\n"); return 1; }
    size_t num_outputs = atoi(argv[3]);
    if (num_outputs <= 0) { fprintf(stderr, "outputs-per-body must be positive\n"); return 1; }
    Matrix* input = matrix_from_npy_path(argv[4]);
    if (input == NULL) { perror("error reading input"); return 1; }
    if (input->cols != 7) { fprintf(stderr, "input.npy must have 7 columns\n"); return 1; }
    size_t n = input->rows;
    if (n == 0) { fprintf(stderr, "input.npy must have at least 1 row\n"); return 1; }
    size_t num_steps = (size_t)(total_time / time_step + 0.5);
    if (num_steps < num_outputs) { num_outputs = 1; }
    size_t output_steps = num_steps/num_outputs;
    num_outputs = (num_steps+output_steps-1)/output_steps;

    // variables available now:
    //   time_step    number of seconds between each time point
    //   total_time   total number of seconds in the simulation
    //   num_steps    number of time steps to simulate (more useful than total_time)
    //   num_outputs  number of times the position will be output for all bodies
    //   output_steps number of steps between each output of the position
    //   input        n-by-7 Matrix of input data
    //   n            number of bodies to simulate

    // start the clock
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);


    //output matrix has the num_outputs amount of rows and 3*n amount of columns, to save the x, y, z positions of each body
    //output's rows are the amount of times the position will be output for all bodies
    //output's cols are the elements of output, which are the x, y, z positions of each body
    Matrix* output = matrix_create_raw(num_outputs, 3*n);
    if (output == NULL) { perror("error creating output matrix"); return 1; }
    matrix_fill_zeros(output);
    

    // Run simulation for each time step

    //Create the matrix's which keep track of the current and future positions and velocities
    //The loop first calculates the future acceleration based on the current position
    //the loop next updates every objects position based on its current velocity
    //Finally the loop makes the future velocity the current velocity
    //position 0 is the mass, 1 is the x position, 2 is the y position, 3 is the z position, 4 is the x velocity, 5 is the y velocity, 6 is the z velocity
    double* masses = (double*)malloc(n * sizeof(double));
    double* current_x = (double*)malloc(n * sizeof(double));
    double* current_y = (double*)malloc(n * sizeof(double));
    double* current_z = (double*)malloc(n * sizeof(double));

    double* velocity_x = (double*)malloc(n * sizeof(double));
    double* velocity_y = (double*)malloc(n * sizeof(double));
    double* velocity_z = (double*)malloc(n * sizeof(double));


    //creates a a forces matrix, which keeps track of the forces acting on each body
    //force[i][j] is the force acting on body i due to body j
    //Matrix* forces = matrix_create_raw(n, n-1);
    //matrix_fill_zeros(forces);
    double* forces = (double*)malloc(n * 3 * sizeof(double));
    for(int i = 0; i < n; i++)
    {   
        masses[i] = MATRIX_AT(input, i, 0);

        current_x[i] = MATRIX_AT(input, i, 1);
        current_y[i] = MATRIX_AT(input, i, 2);
        current_z[i] = MATRIX_AT(input, i, 3);

        velocity_x[i] = MATRIX_AT(input, i, 4);
        velocity_y[i] = MATRIX_AT(input, i, 5);
        velocity_z[i] = MATRIX_AT(input, i, 6);
    }

    //save positions to row `0` of output
    for(size_t i = 0; i < n; i++)
    {
        //this code can be refactored later to use getters
        output->data[i*3] = current_x[i];
        output->data[i*3+1] = current_y[i];
        output->data[i*3+2] = current_z[i];
    }

    
/**
 * 
 * to compute the time step we follow these steps
 * for each object:
 * 1. Calculate  all the gravitational forces on the object due to all other objects using newtons law of gravitation
 * 2. Calculate the net force on the object using the superposition principle with all the forces calculated in step 1
 * 3. Calculate the acceleration of the object using Newton's Second Law of Motion
*/

    //Point totalForces;
    for (size_t t = 1; t < num_steps; t++) 
    {
        //calculate the forces acting on each body
        memset(forces, 0, n * 3 * sizeof(double));
        calculateForcesMatrix(masses, current_x, current_y, current_z, forces, n);
        #pragma omp parallel for shared(velocity_x, velocity_y, velocity_z, forces, masses, current_x, current_y, current_z) 
        for (int i = 0; i < n; i++)
        {
            
            Point acceleration = calculateAcceleration((Point){forces[i*3], forces[i*3+1], forces[i*3+2]}, masses[i]);

            //updates the future velocity based on the acceleration
            velocity_x[i] += acceleration.x * time_step;
            velocity_y[i] += acceleration.y * time_step;
            velocity_z[i] += acceleration.z * time_step;

        }

        //updates the current position and velocity of every body

        for(int i = 0; i < n; i++)
        {
            //updates current velocity based on new velocity
            current_x[i] += velocity_x[i] * time_step;
            current_y[i] += velocity_y[i] * time_step;
            current_z[i] += velocity_z[i] * time_step;
            

        }

        // Periodically copy the positions to the output data
            if (t % output_steps == 0) 
            {            
                for(int j = 0; j < n; j++)
                {   
                    MATRIX_AT(output, t/output_steps, j*3) = current_x[j];
                    MATRIX_AT(output, t/output_steps, j*3+1) = current_y[j];
                    MATRIX_AT(output, t/output_steps, j*3+2) = current_z[j];
                }
            }
    }
    // Save the final set of data if necessary
    if (num_steps % output_steps != 0) 
    {
        for(int j = 0; j < n; j++)
        {
            MATRIX_AT(output, num_outputs-1, j*3) = current_x[j];
            MATRIX_AT(output, num_outputs-1, j*3+1) = current_y[j];
            MATRIX_AT(output, num_outputs-1, j*3+2) = current_z[j];
        }
        

    }


    // get the end and computation time
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time = get_time_diff(&start, &end);
    printf("%f secs\n", time);

    // save results
    matrix_to_npy_path(argv[5], output);

    // cleanup

    free(masses);
    free(current_x);
    free(current_y);
    free(current_z);

    free(velocity_x);
    free(velocity_y);
    free(velocity_z);

    free(output);
    free(input);
        

    return 0;
}



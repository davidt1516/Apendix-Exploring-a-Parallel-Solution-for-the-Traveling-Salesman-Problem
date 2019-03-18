/**
 *   \file tsp-bf.c
 *   \brief Very simple brute-force algorithm implementation to solve TSP
 *  
 *   This program implements the brute-force algorithm to solve the 
 *   Traveling Salesman Problem. This implementation recursively computes
 *   all possible permutations (Hamiltonian cycles or tour) for a given 
 *   vector. The cost associated to a given permutation is computed and 
 *   the minimum cost is updated to a global variable.
 *   
 *   compile: gcc -Wall -fopenmp -o tsp-bf tsp-bf.c
 *   
 *   \author: Danny Munera (2018)
 *            Parallel Programming course Universidad de Antioquia
 * 
 *   Disclaimer: this implementation was partialy based on this code:
 *  http://wikistack.com/traveling-salesman-problem-brute-force-and-dynamic-programming/ 
 *
 *  TODO: print best tour vector 
 *
 */

#include <stdio.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

//#define DEBUG
//#define DEBUG_Dist
#define DEBUG_lim

// global variable containing the minimum cost found
int min_cost = INT_MAX;
// global variable pointing to best tour found
int* best_tour;
int maxfac;

void swap (int *x, int *y);
int compute_cost (int *tour, int* dist, int n);
void brute_force (int *tour, int* dist, int start, int end, int max);
void brute_force_seq (int *tour, int * dist, int start, int end);
int findlimit(int n);

int main(int argc, char *argv[]) {
  int size = 4;
  long seed;
  int i, j;
  double time;
  
  if (argc < 2 && argc > 4){
    printf("Usage: %s <size> <seed> <maxfac>", argv[0]);
  }

  size = strtol(argv[1], NULL, 10);
  seed = argc==3 ? strtol(argv[2], NULL, 10) : 1L;
  maxfac = strtol(argv[3], NULL, 10);
  //random number seed
  srand(seed);

  //Distance matrix initialization
  int* dist = (int*) malloc( sizeof(int) * size * size);
  for (i = 0; i < size; i++){
    for (j = 0; j < size; j++){
      if(i == j){
	// accessing matrix throught vector notation
	// dit[i][j] => dist[i*size+j]
	dist[i*size+j] = 0;
      }else{
	dist[i*size+j] = rand() % 10;
      }
    }
  }
  
#ifdef DEBUG_Dist
  printf("Dist matrix:\n");
  for (i = 0; i < size; i++){
    for (j = 0; j < size; j++){
      printf("%d ",dist[i*size+j]);
    }
    printf("\n");
  }
  printf("\n");
#endif // DEBUG

  //best_tour vector initialization
  best_tour = (int *) malloc (sizeof(int) * size);

  //tour vector
  int * tour = (int *)malloc(sizeof(int) * size);
  //tour initialization
  for(i = 0; i < size; i++){
    tour[i] = i;
  }

  time = omp_get_wtime();
  #pragma omp parallel
  {
    #pragma omp single
    {
      int max;
      max = findlimit(size);
      brute_force(tour, dist, 0, size - 1,max);
    }
  }
  
  time = omp_get_wtime() - time;

  printf("minimum cost %d\ntour: ", min_cost);
  for (i = 0; i < size; i++)
    printf("%d ", best_tour[i]);
  printf("\nExecution time: %.6f seconds\n",time);
  
  return 0;
}

/**
 *  \brief swap
 *  Swap values form two given positions on a vector 
 *  \param x    
 *  \param y 
 */
void swap (int *x, int *y) {
  int temp;
  temp = *x;
  *x = *y;
  *y = temp;
}

/**
 *  \brief compute_cost
 *  Compute cost of a given TSP tour vector
 *  \param tour: pointer to the tour vector
 *  \param dist: pointer to the distance matrix
 *  \param size: size of the tour vector 
 *  \return cost: cost of the tour
 */
int compute_cost (int *tour, int *dist, int size){
  int i, cost = 0, index;
  for (i = 0; i < size; i++) {
    // distance between cities i and i+1
    // dist[ tour[i%size] ] [ tour[(i+1)%size] ]
    index = tour[i % size] * size + tour[(i+1) % size];
    cost += dist[index];
  }
#ifdef DEBUG
  printf("Tour ");
  for(i = 0; i < size;i++){
    printf("%d ", tour[i]);
  }
  printf("cost = %d\n",cost);
#endif // DEBUG
  return cost;
}

/**
 *  \brief brute_force
 *  Brute force algorithm to solve the TSP problem
 *  This function recursively computes all possible permutations for
 *  a given TSP instance and updates the minimum cost and best vector to 
 *  global variables (min_cost, best_tour)
 *  \param tour pointer to the tour vector}
 *  \param dist pointer to the distance matrix
 *  \param start tour vector initial position 
 *  \param end tour vector last position 
 */
void brute_force (int *tour, int * dist, int start, int end, int max){

  int i, cost;
  
  if (start<max){/*Aglomeration Clause*/
    #pragma omp task default(none) firstprivate(start,end,tour,min_cost,best_tour) private(i,cost) shared(max,dist)
    {

      int *tour_deepening;

      if(start == end){
        // Compute cost of each permution
        cost = compute_cost (tour, dist, end+1);
        if (min_cost > cost){
          // Best solution found - copy cost and tour
          min_cost = cost;
          memcpy(best_tour, tour, sizeof(int)*(end+1));
        }
      }else{
        for (i = start; i <= end; i++){
          swap (&tour[start], &tour[i]);
          tour_deepening = (int *) malloc (10*(sizeof(int) * end+1));
          memcpy(tour_deepening, tour, sizeof(int)*(end+1));
          brute_force (tour_deepening, dist, start + 1, end,max);
          //free(tour_deepening);
          swap (&tour[start], &tour[i]);
        }
      }
      free(tour);
    }
  }else{
    if(start == end){
      // Compute cost of each permution
      cost = compute_cost (tour, dist, end+1);
      if (min_cost > cost){
        // Best solution found - copy cost and tour
        min_cost = cost;
        memcpy(best_tour, tour, sizeof(int)*(end+1));
      }
    }else{
      for (i = start; i <= end; i++){
        swap (&tour[start], &tour[i]);
        brute_force_seq(tour, dist, start + 1, end);
        swap (&tour[start], &tour[i]);
      }
    }    
  }
}

void brute_force_seq (int *tour, int * dist, int start, int end){

  int i, cost;
  if(start == end){
    // Compute cost of each permution
    cost = compute_cost (tour, dist, end+1);
    if (min_cost > cost){
      // Best solution found - copy cost and tour
      min_cost = cost;
      memcpy(best_tour, tour, sizeof(int)*(end+1));
    }
  }else{
    for (i = start; i <= end; i++){
      swap (&tour[start], &tour[i]);
      brute_force_seq (tour, dist, start + 1, end);
      swap (&tour[start], &tour[i]);
    }
  }
}

int findlimit(int size){


  int nthreads; 
  int max,i;
  int nfact = 1,maxfact = 1;
  nthreads= omp_get_num_threads();
  int divide;
  i = 2;
  while (i <= size){
    nfact = nfact * i;
    i = i + 1;
  }


  i = size;
  divide = nfact;
  maxfact = 1;
  max = 0;
  while(divide>nthreads*maxfac){
    max = max + 1;
    maxfact = maxfact*i;
    divide = nfact/maxfact;
    i = i - 1; 
  }
#ifdef DEBUG_lim 
  printf("n! %d\n",nfact);
  printf("divide/work per core %d, max %d, threads %d \n",divide,max,nthreads );
#endif

  return max;

}
#include <iostream>
#include <chrono>
#include <cmath>
using namespace std;

int main(int argc, char* argv[]) {
    auto begin = std::chrono::steady_clock::now();

    double tol = atof(argv[1]);
    int size = atoi(argv[2]), iter_max = atoi(argv[3]);
    
    double* A = new double[size*size];
    double* Anew = new double[size*size];
    int iter = 0;
    double error = 1.0;
    double add = 10.0 / (size - 1);

    #pragma acc enter data copyin(A[0:(size * size)], Anew[0:(size * size)])
    #pragma acc kernels
    {
    A[0] = 10;
    A[size - 1] = 20;
    A[(size - 1)*(size)] = 20;
    A[(size - 1)*(size)+ size - 1] = 30;
	for (size_t i = 1; i < size - 1; i++) {
		A[i] = A[i - 1] + add;
        A[(size - 1)*(size)+i] = A[(size - 1)*(size)+i - 1] + add;
        A[i*(size)] = A[(i - 1) *(size)] + add;
        A[i*(size)+size - 1] = A[(i - 1)*(size)+size - 1] + add;
        Anew[i] = A[i - 1] + add;
        Anew[(size - 1)*(size)+i] = A[(size - 1)*(size)+i - 1] + add;
        Anew[i*(size)] = A[(i - 1) *(size)] + add;
        Anew[i*(size)+size - 1] = A[(i - 1)*(size)+size - 1] + add;
	}
    }
    #pragma acc data create(error)
    {
    while ((error > tol) && (iter < iter_max)) {
        iter = iter + 1;
        if ((iter % 100 == 0) or (iter == iter_max) or (iter==1)) {
            #pragma acc kernels async
            {
            error = 0.0;
            }
            #pragma acc parallel num_workers(64) vector_length(16) async
            {
            #pragma acc loop independent collapse(2) reduction(max:error)
            for (int j = 1; j < size - 1; j++) {
                for (int i = 1; i < size - 1; i++) {
                    Anew[i * size + j] = 0.25 * (A[(i + 1) * size + j] + A[(i - 1) * size + j] + A[i * size + j - 1] + A[i * size + j + 1]);
                    error = fmax(error, fabs(Anew[i * size + j] - A[i * size + j]));
                }
            }
        }   
        }
        else{
        #pragma acc parallel num_workers(64) vector_length(16) async
        {
            #pragma acc loop independent collapse(2)
            for (int j = 1; j < size - 1; j++) {
                for (int i = 1; i < size - 1; i++) {
                    Anew[i * size + j] = 0.25 * (A[(i + 1) * size + j] + A[(i - 1) * size + j] + A[i * size + j - 1] + A[i * size + j + 1]);
                }
            }
        }   
        }
        double* swap = A;
		A = Anew;
		Anew = swap;
        if ((iter % 100 == 0) or (iter == iter_max) or (iter==1)) {
            #pragma acc update host(error) wait
            std::cout << iter << ":" << error << "\n";
        }
    }
    }
    #pragma acc exit data delete(A[0:(size * size)], Anew[0:(size * size)], error)
    delete[] A;
    delete[] Anew;

    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end-begin);
    std::cout << "The time:" << elapsed_ms.count() << "ms\n";
    return 0;
}
#include <cublas_v2.h>
#include <iostream>
#include <chrono>
#include <cmath>
using namespace std;
int main() {
    auto begin = std::chrono::steady_clock::now();

    double tol = atof(argv[1]);
    int size = atoi(argv[2]), iter_max = atoi(argv[3]);

    cublasHandle_t handle;
    cublasCreate(&handle);

    const double alpha = -1;
    double* A = new double[size*size];
    double* Anew = new double[size*size];

    int iter = 0;
    int index_max = 0;
    double error = 1.0;
    double add = 10.0 / (size - 1);

    #pragma acc enter data create(A[0:size*size], Anew[0:size*size])
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

    while ((error > tol) && (iter < iter_max)) {
        iter = iter + 1;

        #pragma acc parallel num_workers(64) vector_length(16) async
        {
            #pragma acc loop independent collapse(2)
            for (int i = 1; i < size - 1; i++) {
                for (int j = 1; j < size - 1; j++) {
                    Anew[i * size + j] = 0.25 * (A[(i + 1) * size + j] + A[(i - 1) * size + j] + A[i * size + j - 1] + A[i * size + j + 1]);
                }
            }
        }
        if ((iter % 100 == 0) or (iter == iter_max) or (iter==1)) {
            #pragma acc wait
            #pragma acc host_data use_device(A, Anew)
            cublasDaxpy(handle, size * size, &alpha, Anew, 1, A, 1);
            #pragma acc host_data use_device(A)
            cublasIdamax(handle, size * size, A, 1, &index_max);
            #pragma acc update self(A[index_max-1])
            error = fabs(A[index_max - 1]);
            #pragma acc host_data use_device(A, Anew)
            cublasDcopy(handle, size * size, Anew, 1, A, 1);
        }

        double* swap = A;
        A = Anew;
        Anew = swap;

    }

    if(iter!=iter_max){
        std::cout << iter_max << ":" << error << "\n";
    }
    else{
        std::cout << iter << ":" << error << "\n";
    }

    #pragma acc exit data delete(A[0:(size * size)], Anew[0:(size * size)])
    delete[] A;
    delete[] Anew;

    cublasDestroy(handle);

    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end-begin);
    std::cout << "The time:" << elapsed_ms.count() << "ms\n";
    return 0;
}

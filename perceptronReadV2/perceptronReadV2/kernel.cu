#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

const string training_image_fn = "t10k-images.idx3-ubyte";
const string training_label_fn = "t10k-labels.idx1-ubyte";

const int nTraining = 10000;
const int height = 28;
const int width = 28;
const int neuronas = 10;

ifstream image;
ifstream label;

vector<double> input((height* width) + 1);
vector<double> weights((height* width + 1)* neuronas, 0.0);
int errors = 0;

__global__ void addKernel(const double* d_input, const double* d_weights, double* d_y, int height, int width) {
    int neuron = blockIdx.x;
    int i = threadIdx.x;

    __shared__ double sharedSum[785];
    sharedSum[i] = d_input[i] * d_weights[neuron * (height * width + 1) + i];
    __syncthreads();

    // Reduce sum within the block
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (i < stride) {
            sharedSum[i] += sharedSum[i + stride];
        }
        __syncthreads();
    }

    if (i == 0) {
        d_y[neuron] = sharedSum[0];
        if (d_y[neuron] > 0.0) {
            d_y[neuron] = 1.0;
        }
        else {
            d_y[neuron] = 0.0;
        }
    }
}

int main() {
    std::string filename = "weights.txt";
    std::ifstream inFile(filename);
    if (!inFile) {
        cerr << "No se pudo abrir el archivo para leer." << std::endl;
    }
    else {
        for (size_t i = 0; i < weights.size(); ++i) {
            inFile >> weights[i];
        }
        inFile.close();
    }

    image.open(training_image_fn.c_str(), ios::in | ios::binary);
    label.open(training_label_fn.c_str(), ios::in | ios::binary);

    char number;
    for (int i = 1; i <= 16; ++i) {
        image.read(&number, sizeof(char));
    }
    for (int i = 1; i <= 8; ++i) {
        label.read(&number, sizeof(char));
    }

    //int cantidad = 1;
    //while (cantidad > 0) {
        for (int k = 0; k < nTraining; k++) {
            vector<double> labels(neuronas, 0.0);
            char n;
            for (int i = 0; i < ((height * width) + 1); i++) {
                if (i == 0) {
                    input[0] = 1.0;
                }
                else {
                    image.read(&n, sizeof(char));
                    if (n == 0)
                        input[i] = 0.0;
                    else
                        input[i] = 1.0;
                }
            }
            label.read(&n, sizeof(char));
            labels[n] = 1.0;

            vector<double> y(neuronas, 0.0);
            double* d_weights;
            double* d_input;
            double* d_y;
            double* d_labels;
            cudaMalloc((void**)&d_input, ((height * width) + 1) * sizeof(double));
            cudaMalloc((void**)&d_weights, weights.size() * sizeof(double));
            cudaMalloc((void**)&d_y, neuronas * sizeof(double));
            cudaMalloc((void**)&d_labels, neuronas * sizeof(double));

            cudaMemcpy(d_input, input.data(), ((height * width) + 1) * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(d_weights, weights.data(), weights.size() * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(d_labels, labels.data(), neuronas * sizeof(double), cudaMemcpyHostToDevice);

            addKernel << <neuronas, (height * width) + 1 >> > (d_input, d_weights, d_y, height, width);
            cudaDeviceSynchronize();
            cudaMemcpy(y.data(), d_y, neuronas * sizeof(double), cudaMemcpyDeviceToHost);

            bool converged = true;
            for (int i = 0; i < neuronas; i++) {
                if (y[i] != labels[i]) {
                        converged = false;
                        break;
                }
            }
            if (!converged)
                errors++;
                

            cudaFree(d_weights);
            cudaFree(d_input);
            cudaFree(d_y);
            cudaFree(d_labels);
            cout << ": " << (double)k / (double)nTraining * 100 << "%" << endl;
        }
        //cantidad--;
    //}

        cout << "Cantidad de errores: " << errors<<endl;

    return 0;
}

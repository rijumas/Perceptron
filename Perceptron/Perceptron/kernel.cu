#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

const string training_image_fn = "train-images.idx3-ubyte";
const string training_label_fn = "train-labels.idx1-ubyte";

const int nTraining = 60000;

const int height = 28;
const int width = 28;

const int neuronas = height * width;

ifstream image;
ifstream label;

vector<double> input((height* width) + 1);
vector<double> weights(784 * 785,0.0);

__global__ void addKernel(const double* d_input, const double* d_weights, double* d_y) {
    int i = threadIdx.x;
    d_y[i] = d_input[i]*d_weights[i];
}

__global__ void addKernelPropagation(double* d_input, double* d_weights, double d_y, double d_l) {
    int i = threadIdx.x;
    d_weights[i] += d_input[i] * 0.5 * (d_l - d_y);
}

int main() {
    image.open(training_image_fn.c_str(), ios::in | ios::binary);
    label.open(training_label_fn.c_str(), ios::in | ios::binary);

    char number;
    for (int i = 1; i <= 16; ++i) {
        image.read(&number, sizeof(char));
    }
    for (int i = 1; i <= 8; ++i) {
        label.read(&number, sizeof(char));
    }

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
                //cout << input[i] << " ";
                //if (i % 28 == 0)
                    //cout << endl;
            }
        }
        label.read(&n, sizeof(char));
        labels[n] = 1.0;
        vector<double> weightsTemp(height * width + 1);
        for (int i = 0; i < 784; i++) {

            for (int g = i * 784, o = 0; g <= (i + 1) * 784; g++, o++) {
                weightsTemp[o] = weights[g];
            }
            double u = 0.0;
            vector<double> y(height * width + 1, 1.0);
            double* d_weights;
            double* d_input;
            double* d_y;
            cudaSetDevice(0);
            cudaMalloc((void**)&d_input, ((height * width) + 1) * sizeof(double));
            cudaMalloc((void**)&d_weights, ((height * width) + 1) * sizeof(double));
            cudaMalloc((void**)&d_y, ((height * width) + 1) * sizeof(double));
            cudaMemcpy(d_input, input.data(), ((height * width) + 1) * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(d_weights, weightsTemp.data(), ((height * width) + 1) * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(d_y, y.data(), ((height * width) + 1) * sizeof(double), cudaMemcpyHostToDevice);
            addKernel << <1, ((height * width) + 1) >> > (d_input, d_weights, d_y);
            cudaDeviceSynchronize();
            cudaMemcpy(y.data(), d_y, sizeof(double) * (height * width + 1), cudaMemcpyDeviceToHost);
            cudaMemcpy(weightsTemp.data(), d_weights, sizeof(double) * (height * width + 1), cudaMemcpyDeviceToHost);

            for (int i = 0; i < y.size(); i++) {
                u += y[i];
            }
            if (u > 0)
                u = 1.0;
            else
                u = 0.0;
            while (u != labels[i]) {
                addKernelPropagation << <1, ((height * width) + 1) >> > (d_input, d_weights, u, labels[i]);
                cudaDeviceSynchronize();
                cudaMemcpy(weightsTemp.data(), d_weights, sizeof(double) * (height * width + 1), cudaMemcpyDeviceToHost);
                u = 0.0;
                cudaSetDevice(0);
                cudaMalloc((void**)&d_input, ((height * width) + 1) * sizeof(double));
                cudaMalloc((void**)&d_weights, ((height * width) + 1) * sizeof(double));
                cudaMalloc((void**)&d_y, ((height * width) + 1) * sizeof(double));
                cudaMemcpy(d_input, input.data(), ((height * width) + 1) * sizeof(double), cudaMemcpyHostToDevice);
                cudaMemcpy(d_weights, weightsTemp.data(), ((height * width) + 1) * sizeof(double), cudaMemcpyHostToDevice);
                cudaMemcpy(d_y, y.data(), ((height * width) + 1) * sizeof(double), cudaMemcpyHostToDevice);
                addKernel << <1, ((height * width) + 1) >> > (d_input, d_weights, d_y);
                cudaDeviceSynchronize();
                cudaMemcpy(y.data(), d_y, sizeof(double) * (height * width + 1), cudaMemcpyDeviceToHost);
                cudaMemcpy(weightsTemp.data(), d_weights, sizeof(double) * (height * width + 1), cudaMemcpyDeviceToHost);

                for (int i = 0; i < y.size(); i++) {
                    u += y[i];
                }
                u = (u > 0);
            }
            //cout << "label: " << labels[i] << " u: " << u<<endl;
            for (int g = i*784, o=0;g <= (i+1)*784; g++,o++) {
                weights[g] = weightsTemp[o];
            }
            cudaFree(d_weights);
            cudaFree(d_input);
            cudaFree(d_y);
        }
        system("cls");
        cout << (double)k/(double)nTraining <<"%";
    }
    std::string filename = "weights.txt";

    // Escribir el vector weights en el archivo
    ofstream outFile(filename);
    if (!outFile) {
        cerr << "No se pudo abrir el archivo para escribir." << std::endl;
        return 1;
    }

    for (const double& weight : weights) {
        outFile << weight << "\n";
    }
    outFile.close();
    
    return 0;
}



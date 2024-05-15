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

const int neuronas = height * width;

ifstream image;
ifstream label;

vector<double> input((height* width) + 1);

__global__ void addKernel(const double* d_input, const double* d_weights, double* d_y) {
    int i = threadIdx.x;
    d_y[i] = d_input[i] * d_weights[i];
}
int errores=0;

int main() {
    std::vector<double> weights(784 * 785);
    std::string filename = "weights.txt";
    std::ifstream inFile(filename);
    if (!inFile) {
        cerr << "No se pudo abrir el archivo para leer." << std::endl;
        return 1;
    }

    for (size_t i = 0; i < weights.size(); ++i) {
        inFile >> weights[i];
    }
    inFile.close();
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
            if (u > 0) {
                u = 1.0;
                if (labels[i] != u)
                    errores++;
                //cout <<"label: " <<labels[i]<< "  numero reconocido: " << i << endl;
            }
            else
                u = 0.0;

        }
        system("cls");
        cout << (double)k / (double)nTraining << "%";
    }
    cout << "La cantidad de errores es " << errores;
    return 0;
}



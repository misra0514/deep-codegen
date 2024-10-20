#include "stdio.h"
#include "iostream"

using namespace::std;

__global__ void hello(){
}

int main(){
    hello<<<1,2>>>();
    cout<<cudaGetErrorName(cudaGetLastError());
}
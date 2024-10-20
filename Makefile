libgp.so: kernel.cu kernel.h op.h Makefile
	~/app/cuda-11.8/bin/nvcc -Xptxas -O3 --compiler-options '-fpic -O3 -fopenmp' -gencode=arch=compute_70,code=compute_70 -o libgp.so --shared kernel.cu

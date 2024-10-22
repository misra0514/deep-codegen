#pragma once

#include <string>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

using std::string;
typedef uint32_t vid_t;


class graph_t {
private:
    int * SrcVertex;
    int * DestVertex;
    int * offset;
    int VNumber = 0;
    int ENumber = 0;
public:
    void init(vid_t a_vcount, vid_t a_dstsize, void* a_offset, void* a_nebrs, 
              void* a_offset1, void* a_nebrs1, int64_t a_flag, vid_t edge_count) {
                // This init function was probably made for CSR form. 
              };
    void init(vid_t a_vcount, vid_t edge_count, void * a_srcs ,void* a_nebrs, void* a_offset) {
                // here we only use the following variables:
                // a_vcount:VNumber, a_offset: source array, a_nebrs:dest arrays , edge_count:Enumber
                // while doing init. data need to be transferd to GPU 
                int sizeEdges = edge_count*sizeof(int);
                int sizeOffset = (a_vcount +1)*sizeof(int);
                this->VNumber = a_vcount;
                this->ENumber = edge_count;

                cudaMalloc((void**)(&(this->offset)),sizeOffset);
                cudaMemcpy(this->offset, a_offset, sizeOffset, cudaMemcpyHostToDevice);
                cudaMalloc((void**)(&(this->DestVertex)),sizeEdges);
                cudaMemcpy(this->DestVertex, a_nebrs, sizeEdges, cudaMemcpyHostToDevice);
                cudaMalloc((void**)(&(this->SrcVertex)),sizeEdges);
                cudaMemcpy(this->SrcVertex, a_srcs, sizeEdges, cudaMemcpyHostToDevice);
 
              };
    ~graph_t(){
      cudaFree(SrcVertex);
      cudaFree(DestVertex);
      cudaFree(offset);
    }
    void save_graph(const string& full_path) {};
    void load_graph(const string& full_path) {};
    void load_graph_noeid(const string& full_path) {};
    int get_vcount() {return this->VNumber;};
    int get_ecount() {return this->ENumber;};
};

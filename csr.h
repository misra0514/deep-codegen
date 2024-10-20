#pragma once

#include <string>
#include <iostream>
#include <vector>

using std::string;
typedef uint32_t vid_t;


class graph_t {
private:

public:
    int * SourceVertex;
    int * TargetVertex;
    int VNumber = 0;
    int ENumber = 0;
    void init(vid_t a_vcount, vid_t a_dstsize, void* a_offset, void* a_nebrs, 
              void* a_offset1, void* a_nebrs1, int64_t a_flag, vid_t edge_count) {
                // This init function was probably made for CSR form. 
              };
    void init(vid_t a_vcount, vid_t edge_count, void* a_offset, void* a_nebrs) {
                // here we only use the following variables.
                // a_vcount:VNumber, a_offset: source array, a_nebrs:dest arrays , edge_count:Enumber
                this->VNumber = a_vcount;
                this->ENumber = edge_count;
                this->SourceVertex = (int*)(a_offset);
                this->TargetVertex = (int*)(a_nebrs);
                // std::cout<<this->VNumber<<std::endl;
              };
    void save_graph(const string& full_path) {};
    void load_graph(const string& full_path) {};
    void load_graph_noeid(const string& full_path) {};
    int get_vcount() {return this->VNumber;};
    int get_ecount() {return this->ENumber;};
};

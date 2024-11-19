#pragma once
#include "csr.h"
#include "op.h"

void linear(array2d_t<float>& X, array2d_t<float>& W,  array2d_t<float>& output1);
void gspmmv(graph_t& graph, array2d_t<float>& input1, array2d_t<float>& output, bool reverse, bool norm, uintptr_t stream_handle);
inline void export_kernel(py::module &m) { 
    m.def("linear",[](py::capsule& X, py::capsule& W, py::capsule& output1){
        array2d_t<float> X_array = capsule_to_array2d(X);
        array2d_t<float> W_array = capsule_to_array2d(W);
        array2d_t<float> output1_array = capsule_to_array2d(output1);
    return linear(X_array, W_array, output1_array);
    }
  );
    m.def("gspmmv",[](graph_t& graph, py::capsule& input1, py::capsule& output, bool reverse, bool norm, uintptr_t stream_handle){
        array2d_t<float> input1_array = capsule_to_array2d(input1);
        array2d_t<float> output_array = capsule_to_array2d(output);
    return gspmmv(graph, input1_array, output_array, reverse, norm, stream_handle);
    }
  );
}
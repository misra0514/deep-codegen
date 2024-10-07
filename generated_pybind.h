inline void export_kernel(py::module &m) { 
    m.def("linear",[](py::capsule& X, py::capsule& W, py::capsule& output1){
        array2d_t<float> X_array = capsule_to_array2d(X);
        array2d_t<float> W_array = capsule_to_array2d(W);
        array2d_t<float> output1_array = capsule_to_array2d(output1);
    return linear(X_array, W_array, output1_array);
    }
  );
}
#include "src/include/tensors.h"
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <iostream>

using namespace simple_cuda;
int main(){
    Extent<2> tensor_extents({2, 10});
    std::cout<<"The tensor index for {1,2}: "<<tensor_extents.index(1, 2)<<std::endl;
    std::cout <<"The tensor index for {0,0}: "<<tensor_extents.index(0, 0)<<std::endl;
    std::cout<<"The tensor index for {1,0}: "<<tensor_extents.index(1, 0)<<std::endl;
    // fmt::print("The tensor index for {1,2}: {}\n", tensor_extents.index(1, 2));
    // fm
    HostTensor<float, Extent<2>> h_tensor(tensor_extents);

    // fmt::print("The size of the tensor is {}\n", h_tensor.data_.size());
    std::cout<<"The size of the tensor is "<<h_tensor.data_.size()<<std::endl;
    for(const auto& val: h_tensor.data_){
        std::cout<<val<<" ";
    }
    std::cout<<std::endl;

    DeviceTensor<float, Extent<2>> d_tensor(tensor_extents);
    std::cout<<"The size of the tensor is "<<d_tensor.data_.size()<<std::endl;


    // Lets try the copy constructurs
    auto d_to_h = d_tensor.to_host();
    auto h_to_d = h_tensor.to_device();

    std::cout<<"The size of the tensor is "<<d_to_h.data_.size()<<std::endl;
    std::cout<<"The size of the tensor is "<<h_to_d.data_.size()<<std::endl;

    for(const auto& val: d_to_h.data_){
        std::cout<<val<<" ";
    }


    return 0;
}
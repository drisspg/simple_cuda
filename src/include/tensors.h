#pragma once
#include <array>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace simple_cuda {

template <int num_dims> struct Extent {

  Extent(std::array<size_t, num_dims> size) : size_(size) {
    size_t stride = 1;
    for (int i = num_dims - 1; i >= 0; i--) {
      stride_[i] = stride;
      stride *= size[i];
    }
  }
  std::array<size_t, num_dims> size_;
  std::array<size_t, num_dims> stride_;

  size_t n_dim_ = num_dims;

  template <typename... Args> __host__ size_t index(Args... args) const {
    static_assert(sizeof...(args) == num_dims,
                  "Number of arguments must be equal to n_dim");

    size_t index = 0;
    std::initializer_list<int> indices{args...};
    auto it = indices.begin();
    for (int i = 0; i < num_dims; i++) {
      index += (*it) * stride_[i];
      ++it;
    }

    return index;
  }
};

template <typename T, size_t n_dim, typename ExtentType> struct HostTensor {
public:
  HostTensor(int size, ExtentType extent) : extent_(extent) {
    data_ = thrust::host_vector<float>(size);
  }

  T *data_ptr() { return thrust::raw_pointer_cast(data_.data()); }

  thrust::host_vector<T> data_;
  ExtentType extent_;
};

template <typename T, size_t n_dim, typename ExtentType> struct DeviceTensor {
  DeviceTensor(int size, ExtentType extent) : extent_(extent) {
    data_ = thrust::device_vector<float>(size);
  }

  T *data_ptr() { return thrust::raw_pointer_cast(data_.data()); }

  thrust::device_vector<T> data_;
  ExtentType extent_;
};

} // namespace simple_cuda

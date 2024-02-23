#pragma once
#include <array>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace simple_cuda {

// Forward declaration of DeviceTensor
template <typename T, typename ExtentType> struct DeviceTensor;

template <int num_dims> struct Extent {
  Extent() = default;
  Extent(std::array<size_t, num_dims> size) : size_(size) {
    size_t stride = 1;
    for (int i = num_dims - 1; i >= 0; i--) {
      stride_[i] = stride;
      stride *= size[i];
    }
  }
  std::array<size_t, num_dims> size_;
  std::array<size_t, num_dims> stride_;

  static constexpr int n_dim = num_dims;

  size_t numel() const {
    size_t numel = 1;
    for (int i = 0; i < num_dims; i++) {
      numel *= size_[i];
    }
    return numel;
  }
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

template <typename T, typename ExtentType> struct HostTensor {
  HostTensor(int size, ExtentType extent) : extent_(extent) {
    data_ = thrust::host_vector<float>(size);
  }
  HostTensor(ExtentType extent) : extent_(extent) {
    size_t size = 1;
    for (int i = 0; i < ExtentType::n_dim; i++) {
      size *= extent.size_[i];
    }
    data_ = thrust::host_vector<float>(size);
  }

  HostTensor(const DeviceTensor<T, ExtentType> &device_tensor) {
    extent_ = device_tensor.extent_;
    data_ = device_tensor.data_;
  }

  DeviceTensor<T, ExtentType> to_device() {
    return DeviceTensor<T, ExtentType>(*this);
  }

  T *data_ptr() { return thrust::raw_pointer_cast(data_.data()); }

  thrust::host_vector<T> data_;
  ExtentType extent_;
};

template <typename T, typename ExtentType> struct DeviceTensor {
  DeviceTensor(int size, ExtentType extent) : extent_(extent) {
    data_ = thrust::device_vector<float>(size);
  }
  DeviceTensor(ExtentType extent) : extent_(extent) {
    size_t size = 1;
    for (int i = 0; i < ExtentType::n_dim; i++) {
      size *= extent.size_[i];
    }
    data_ = thrust::device_vector<float>(size);
  }

  DeviceTensor(const HostTensor<T, ExtentType> &host_tensor) {
    extent_ = host_tensor.extent_;
    data_ = host_tensor.data_;
  }

  HostTensor<T, ExtentType> to_host() {
    return HostTensor<T, ExtentType>(*this);
  }

  T *data_ptr() { return thrust::raw_pointer_cast(data_.data()); }

  thrust::device_vector<T> data_;
  ExtentType extent_;
};

} // namespace simple_cuda

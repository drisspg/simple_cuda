#pragma once
#include <array>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <fmt/core.h>
#include <fmt/ranges.h>

namespace simple_cuda {

// Forward declaration of DeviceTensor
template <typename T, typename ExtentType> struct DeviceTensor;

template <int num_dims> struct Extent {
  // Constructors
  Extent() = default;
  Extent(std::array<size_t, num_dims> size) : size_(size) {
    size_t stride = 1;
    for (int i = num_dims - 1; i >= 0; i--) {
      stride_[i] = stride;
      stride *= size[i];
    }
    numel_ = compute_numel();
  }

  // Members
  std::array<size_t, num_dims> size_;
  std::array<size_t, num_dims> stride_;
  static constexpr int n_dim = num_dims;
  size_t numel_;

  // Methods
  void transpose(size_t dim1, size_t dim2) {
    assert((dim1 < num_dims && dim2 < num_dims) &&
           "Dimensions must be less than n_dim");
    std::swap(size_[dim1], size_[dim2]);
    std::swap(stride_[dim1], stride_[dim2]);
  }
  template<size_t dim>
  constexpr size_t size() {
    static_assert(dim < num_dims, "Dimension must be less than n_dim");
    return size_[dim];
  }
  size_t numel() const { return numel_; }
  size_t compute_numel() const {
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
    data_ = thrust::host_vector<float>(extent.numel());
  }

  HostTensor(const DeviceTensor<T, ExtentType> &device_tensor)
      : extent_(device_tensor.extent_), data_(device_tensor.data_) {}

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
    data_ = thrust::device_vector<float>(extent.numel());
  }

  DeviceTensor(const HostTensor<T, ExtentType> &host_tensor)
      : extent_(host_tensor.extent_), data_(host_tensor.data_) {}

  HostTensor<T, ExtentType> to_host() {
    return HostTensor<T, ExtentType>(*this);
  }

  T *data_ptr() { return thrust::raw_pointer_cast(data_.data()); }

  thrust::device_vector<T> data_;
  ExtentType extent_;
};

} // namespace simple_cuda

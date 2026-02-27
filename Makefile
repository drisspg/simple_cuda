.DEFAULT_GOAL := all

build_dir := build
nvcc_path := $(shell command -v nvcc)
detected_compute_cap := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '.')
base_cuda_arch := $(if $(strip $(detected_compute_cap)),$(detected_compute_cap),100)
cuda_arch := $(shell if [ "$(base_cuda_arch)" -ge 90 ]; then printf "%sa" "$(base_cuda_arch)"; else printf "%s" "$(base_cuda_arch)"; fi)

ifeq ($(strip $(nvcc_path)),)
$(error nvcc not found in PATH. Set CUDACXX or update PATH to point to your CUDA toolkit)
endif

all: release

release:
	cmake -B$(build_dir) -GNinja -DCMAKE_CUDA_COMPILER=$(nvcc_path) -DCMAKE_CUDA_ARCHITECTURES=$(cuda_arch) -DSIMPLE_CUDA_ARCH=$(cuda_arch) -DCMAKE_BUILD_TYPE=RelWithDebInfo
	ninja -C $(build_dir)

debug:
	cmake -B$(build_dir) -GNinja -DCMAKE_CUDA_COMPILER=$(nvcc_path) -DCMAKE_CUDA_ARCHITECTURES=$(cuda_arch) -DSIMPLE_CUDA_ARCH=$(cuda_arch) -DCMAKE_BUILD_TYPE=Debug
	ninja -C $(build_dir)

clean:
	rm -rf $(build_dir)

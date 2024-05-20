.DEFAULT_GOAL := all

build_dir := build

all: release

release:
	cmake -B$(build_dir) -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo
	ninja -C $(build_dir)

debug:
	cmake -B$(build_dir) -GNinja -DCMAKE_BUILD_TYPE=Debug
	ninja -C $(build_dir)

clean:
	rm -rf $(build_dir)

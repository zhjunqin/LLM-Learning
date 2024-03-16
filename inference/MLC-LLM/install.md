# 两种安装方式

## From pip

从 https://mlc.ai/wheels 下载对应的 PIP 包，然后安装。

或者 pip 安装

```
python3 -m pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly mlc-ai-nightly
```

## Installed from source

https://llm.mlc.ai/docs/install/tvm.html

### Install TVM

#### 依赖 

```
CMake >= 3.24
LLVM >= 15
Git
(Optional) CUDA >= 11.8 (targeting NVIDIA GPUs)
(Optional) Metal (targeting Apple GPUs such as M1 and M2)
(Optional) Vulkan (targeting NVIDIA, AMD, Intel and mobile GPUs)
(Optional) OpenCL (targeting NVIDIA, AMD, Intel and mobile GPUs)
```

- CMake 安装
```
ARCH=$(uname -m)
CMAKE_VERSION="3.24.4"

PARSED_CMAKE_VERSION=$(echo $CMAKE_VERSION | sed 's/\.[0-9]*$//')
CMAKE_FILE_NAME="cmake-${CMAKE_VERSION}-linux-${ARCH}"
RELEASE_URL_CMAKE=https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/${CMAKE_FILE_NAME}.tar.gz
wget --no-verbose ${RELEASE_URL_CMAKE} -P /tmp
tar -xf /tmp/${CMAKE_FILE_NAME}.tar.gz -C /usr/local/
ln -s /usr/local/${CMAKE_FILE_NAME} /usr/local/cmake

export PATH=/usr/local/cmake/bin:$PATH
```
- LLVM 安装

https://apt.llvm.org/

```
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh <version number>

To install all apt.llvm.org packages at once:
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh <version number> all

示例：
sudo ./llvm.sh 15 all # 有时候必须用 all，不然会缺少一些库，比如 /usr/lib/llvm-15/lib/libPolly.a
```

- 代码编译

```
# clone from GitHub
git clone --recursive git@github.com:mlc-ai/relax.git tvm-unity && cd tvm-unity
# create the build directory
rm -rf build && mkdir build && cd build
# specify build requirements in `config.cmake`
cp ../cmake/config.cmake .
```

配置 config.cmake

```
# controls default compilation flags
echo "set(CMAKE_BUILD_TYPE RelWithDebInfo)" >> config.cmake
# LLVM is a must dependency
echo "set(USE_LLVM \"llvm-config --ignore-libllvm --link-static\")" >> config.cmake
echo "set(HIDE_PRIVATE_SYMBOLS ON)" >> config.cmake
# GPU SDKs, turn on if needed
echo "set(USE_CUDA   OFF)" >> config.cmake
echo "set(USE_METAL  OFF)" >> config.cmake
echo "set(USE_VULKAN OFF)" >> config.cmake
echo "set(USE_OPENCL OFF)" >> config.cmake
# FlashInfer related, requires CUDA w/ compute capability 80;86;89;90
echo "set(USE_FLASHINFER OFF)" >> config.cmake
echo "set(FLASHINFER_CUDA_ARCHITECTURES YOUR_CUDA_COMPUTE_CAPABILITY_HERE)" >> config.cmake
echo "set(CMAKE_CUDA_ARCHITECTURES YOUR_CUDA_COMPUTE_CAPABILITY_HERE)" >> config.cmake
```

Jetson 示例
```
# Set Windows Visual Studio default Architecture (equivalent to -A x64)
SET(CMAKE_VS_PLATFORM_NAME_DEFAULT "aarch64")

# Set Windows Visual Studio default host (equivalent to -Thost=x64)
SET(CMAKE_VS_PLATFORM_TOOLSET_HOST_ARCHITECTURE "aarch64")
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(USE_LLVM "llvm-config-15 --ignore-libllvm --link-static")
set(HIDE_PRIVATE_SYMBOLS ON)
set(USE_VLLM ON)
set(USE_CUBLAS   ON)
set(USE_CUDNN   ON)
set(USE_CUTLASS ON)
set(USE_THRUST ON)

set(USE_CUDA   ON)
set(USE_METAL  OFF)
set(USE_VULKAN OFF)
set(USE_OPENCL OFF)

set(USE_FLASHINFER ON)
set(FLASHINFER_CUDA_ARCHITECTURES 87)
set(CMAKE_CUDA_ARCHITECTURES 87)
```

Cuda 11.4

```
# Set Windows Visual Studio default Architecture (equivalent to -A x64)
SET(CMAKE_VS_PLATFORM_NAME_DEFAULT "aarch64")

# Set Windows Visual Studio default host (equivalent to -Thost=x64)
SET(CMAKE_VS_PLATFORM_TOOLSET_HOST_ARCHITECTURE "aarch64")

set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(USE_LLVM "llvm-config-15 --ignore-libllvm --link-static")
set(HIDE_PRIVATE_SYMBOLS ON)
set(USE_CUDA   ON)

set(USE_VLLM ON)
set(USE_CUBLAS   ON)
set(USE_CUDNN   ON)
set(USE_CUTLASS OFF)

set(USE_METAL  OFF)
set(USE_VULKAN OFF)
set(USE_OPENCL OFF)
set(USE_FLASHINFER OFF)
set(FLASHINFER_CUDA_ARCHITECTURES 87)
set(CMAKE_CUDA_ARCHITECTURES 87)
```

```
cmake .. && cmake --build . --parallel $(nproc)

cd /path-to-tvm-unity/python
pip install -e .

```

#### 验证

```
# python -c "import tvm; print('\n'.join(f'{k}: {v}' for k, v in tvm.support.libinfo().items()))"
... # Omitted less relevant options
GIT_COMMIT_HASH: 4f6289590252a1cf45a4dc37bce55a25043b8338
HIDE_PRIVATE_SYMBOLS: ON
USE_LLVM: llvm-config --link-static
LLVM_VERSION: 15.0.7
USE_VULKAN: OFF
USE_CUDA: OFF
CUDA_VERSION: NOT-FOUND
USE_OPENCL: OFF
USE_METAL: ON
USE_ROCM: OFF
```

### Install  MLC

#### 依赖

```
CMake >= 3.24
Git
Rust and Cargo, required by Hugging Face’s tokenizer
One of the GPU runtimes:
CUDA >= 11.8 (NVIDIA GPUs)
Metal (Apple GPUs)
Vulkan (NVIDIA, AMD, Intel GPUs)
```

#### Rust 安装

https://doc.rust-lang.org/cargo/getting-started/installation.html

```
curl https://sh.rustup.rs -sSf | sh

This is usually done by running one of the following (note the leading DOT):
. "$HOME/.cargo/env"            # For sh/bash/zsh/ash/dash/pdksh
source "$HOME/.cargo/env.fish"  # For fish

source $HOME/.cargo/env
```

#### 代码编译

```
git clone --recursive https://github.com/mlc-ai/mlc-llm.git && cd mlc-llm/
# create build directory
mkdir -p build && cd build
# generate build configuration
python3 ../cmake/gen_cmake_config.py

```
配置示例
```
set(TVM_HOME /data/mlc/tvm-unity/)
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(USE_CUDA ON)
set(USE_CUTLASS ON)
set(USE_CUBLAS ON)
set(USE_ROCM OFF)
set(USE_VULKAN OFF)
set(USE_METAL OFF)
set(USE_OPENCL OFF)
set(USE_THRUST ON)
set(USE_FLASHINFER ON)
set(FLASHINFER_CUDA_ARCHITECTURES 87)
set(CMAKE_CUDA_ARCHITECTURES 87)
```

```
# build mlc_llm libraries
cmake .. && cmake --build . --parallel $(nproc) && cd ..

cd /path-to-mlc-llm/python
pip install -e .
```

#### 验证

```
python -c "import mlc_llm; print(mlc_llm)"
```
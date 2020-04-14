# compile with cuda
WITH_GPU=OFF
# compile with tensorrt
WITH_TENSORRT=OFF
# path to paddle inference lib
PADDLE_DIR=/root/projects/deps/fluid_inference/
# path to opencv lib
OPENCV_DIR=$(pwd)/deps/opencv346/
# path to cuda lib
CUDA_LIB=/usr/local/cuda/lib64/
# path to cudnn lib
CUDNN_LIB=/usr/local/cuda/lib64/
# with static or shared library
WITH_STATIC_LIB=OFF

sh $(pwd)/scripts/bootstrap.sh

rm -rf build
mkdir -p build
cd build
make clean
cmake .. \
    -DWITH_GPU=${WITH_GPU} \
    -DWITH_TENSORRT=${WITH_TENSORRT} \
    -DWITH_STATIC_LIB=${WITH_STATIC_LIB} \
    -DPADDLE_DIR=${PADDLE_DIR} \
    -DCUDA_LIB=${CUDA_LIB} \
    -DCUDNN_LIB=${CUDNN_LIB} \
    -DOPENCV_DIR=${OPENCV_DIR}
make

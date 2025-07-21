@echo off
REM RTX 5090 Environment Setup for Windows
REM Generated on July 21, 2025
echo.
echo 🎮 RTX 5090 ENVIRONMENT SETUP
echo ================================
echo Setting up optimal environment for RTX 5090...
echo.
set "FLAGS_fraction_of_gpu_memory_to_use=0.8"
set "FLAGS_conv_workspace_size_limit=512"
set "FLAGS_cudnn_deterministic=true"
set "FLAGS_memory_pool_reuse=true"
set "FLAGS_allocator_strategy=naive_best_fit"
set "CUDA_VISIBLE_DEVICES=0"
set "CUDA_DEVICE_ORDER=PCI_BUS_ID"
set "CUDA_CACHE_DISABLE=0"
set "CUDA_LAUNCH_BLOCKING=0"
set "TF_GPU_ALLOCATOR=cuda_malloc_async"
set "TF_FORCE_GPU_ALLOW_GROWTH=true"
set "TF_GPU_THREAD_MODE=gpu_private"
set "TF_GPU_THREAD_COUNT=8"
set "TF_XLA_FLAGS=--tf_xla_enable_xla_devices"
set "XLA_GPU_JIT=1"
set "XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda"
set "TF_ENABLE_AUTO_MIXED_PRECISION=1"
set "NVIDIA_TF32_OVERRIDE=1"
set "CUDA_PROFILE=1"
set "NVTX_INJECTION64_PATH="
set "OMP_NUM_THREADS=16"
set "MKL_NUM_THREADS=16"
set "PYTHONOPTIMIZE=1"
set "PYTHONHASHSEED=0"
echo.
echo ✅ RTX 5090 environment variables configured!
echo 💾 Available GPU Memory: 19.2GB (80% of 24GB)
echo ⚡ Performance: Optimized for training and inference
echo.
echo To make permanent, run:
echo setx FLAGS_fraction_of_gpu_memory_to_use 0.8
echo setx FLAGS_conv_workspace_size_limit 512
echo setx FLAGS_cudnn_deterministic true
echo.
pause
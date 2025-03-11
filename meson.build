project(
    'MCX hello world',
    'c',
    'cpp',
    version : '0.1',
    default_options : ['warning_level=2', 'c_std=c11', 'cpp_std=c++17']
)

# Define target architecture and toolchain
toolchain = 'arm-none-eabi-'
cc = meson.get_compiler('c')
cpp = meson.get_compiler('cpp')
c_args = ['-lnosys', '--specs=nosys.specs', '-Os', '-ffunction-sections', '-fdata-sections']
cpp_args = ['-lnosys', '--specs=nosys.specs', '-std=c++17', '-Os', '-ffunction-sections', '-fdata-sections', '-fno-rtti', '-Wno-unused-parameter']
ld_args = [
  '-T', '../lib/mcux-sdk/devices/MCXN947/gcc/MCXN947_cm33_core0_flash.ld',
  '-mcpu=cortex-m33', '-mthumb', '-mfloat-abi=soft',
  '-lc', '-lstdc++', '-lnosys', '--specs=nosys.specs',
]


add_project_arguments(
'-DCMSIS_NN',
'-DTFLITE_STRIP_ERROR_STRINGS',
'-DTF_LITE_DISABLE_DEFAULT_REPORT_ERRORS',
'-DTF_LITE_STATIC_MEMORY',
'-DCMSIS_DEVICE_ARM_CORTEX_M_XX_HEADER_FILE="ARMCM33.h"',
'-DTF_LITE_DISABLE_X86_NEON',
language: 'cpp'
)

# Include directories
freertos_inc = include_directories(
    'lib/freertos/include',
    'lib/freertos/portable/GCC/ARM_CM33/non_secure/',
    'lib/freertos/portable/GCC/ARM_CM33/secure',
    'config',
)

tflite_inc = include_directories(
    'lib/tflm-cmsis/',
    'lib/tflm-cmsis/third_party/flatbuffers/include',
    'lib/tflm-cmsis/third_party/gemmlowp',
    'lib/tflm-cmsis/third_party/ruy',
    'lib/tflm-cmsis/third_party/kissfft',
    'model',
    'lib/tflm-cmsis/third_party/cmsis_nn',
)

cmsis_inc = include_directories(
    'lib/cmsis/CMSIS/Core/Include',
    'lib/cmsis/Device/ARM/ARMCM33/Include/'
)

tflite_src = files(
'lib/tflm-cmsis/signal/micro/kernels/delay.cc',
'lib/tflm-cmsis/signal/micro/kernels/energy.cc',
'lib/tflm-cmsis/signal/micro/kernels/fft_auto_scale_common.cc',
'lib/tflm-cmsis/signal/micro/kernels/fft_auto_scale_kernel.cc',
'lib/tflm-cmsis/signal/micro/kernels/filter_bank.cc',
'lib/tflm-cmsis/signal/micro/kernels/filter_bank_log.cc',
'lib/tflm-cmsis/signal/micro/kernels/filter_bank_spectral_subtraction.cc',
'lib/tflm-cmsis/signal/micro/kernels/filter_bank_square_root.cc',
'lib/tflm-cmsis/signal/micro/kernels/filter_bank_square_root_common.cc',
'lib/tflm-cmsis/signal/micro/kernels/framer.cc',
'lib/tflm-cmsis/signal/micro/kernels/irfft.cc',
'lib/tflm-cmsis/signal/micro/kernels/overlap_add.cc',
'lib/tflm-cmsis/signal/micro/kernels/pcan.cc',
'lib/tflm-cmsis/signal/micro/kernels/rfft.cc',
'lib/tflm-cmsis/signal/micro/kernels/stacker.cc',
'lib/tflm-cmsis/signal/micro/kernels/window.cc',
'lib/tflm-cmsis/signal/src/circular_buffer.cc',
'lib/tflm-cmsis/signal/src/energy.cc',
'lib/tflm-cmsis/signal/src/fft_auto_scale.cc',
'lib/tflm-cmsis/signal/src/filter_bank.cc',
'lib/tflm-cmsis/signal/src/filter_bank_log.cc',
'lib/tflm-cmsis/signal/src/filter_bank_spectral_subtraction.cc',
'lib/tflm-cmsis/signal/src/filter_bank_square_root.cc',
'lib/tflm-cmsis/signal/src/irfft_float.cc',
'lib/tflm-cmsis/signal/src/irfft_int16.cc',
'lib/tflm-cmsis/signal/src/irfft_int32.cc',
'lib/tflm-cmsis/signal/src/kiss_fft_wrappers/kiss_fft_float.cc',
'lib/tflm-cmsis/signal/src/kiss_fft_wrappers/kiss_fft_int16.cc',
'lib/tflm-cmsis/signal/src/kiss_fft_wrappers/kiss_fft_int32.cc',
'lib/tflm-cmsis/signal/src/log.cc',
'lib/tflm-cmsis/signal/src/max_abs.cc',
'lib/tflm-cmsis/signal/src/msb_32.cc',
'lib/tflm-cmsis/signal/src/msb_64.cc',
'lib/tflm-cmsis/signal/src/overlap_add.cc',
'lib/tflm-cmsis/signal/src/pcan_argc_fixed.cc',
'lib/tflm-cmsis/signal/src/rfft_float.cc',
'lib/tflm-cmsis/signal/src/rfft_int16.cc',
'lib/tflm-cmsis/signal/src/rfft_int32.cc',
'lib/tflm-cmsis/signal/src/square_root_32.cc',
'lib/tflm-cmsis/signal/src/square_root_64.cc',
'lib/tflm-cmsis/signal/src/window.cc',
'lib/tflm-cmsis/tensorflow/compiler/mlir/lite/core/api/error_reporter.cc',
'lib/tflm-cmsis/tensorflow/compiler/mlir/lite/schema/schema_utils.cc',
'lib/tflm-cmsis/tensorflow/lite/core/api/flatbuffer_conversions.cc',
'lib/tflm-cmsis/tensorflow/lite/core/api/tensor_utils.cc',
'lib/tflm-cmsis/tensorflow/lite/core/c/common.cc',
'lib/tflm-cmsis/tensorflow/lite/kernels/internal/common.cc',
'lib/tflm-cmsis/tensorflow/lite/kernels/internal/portable_tensor_utils.cc',
'lib/tflm-cmsis/tensorflow/lite/kernels/internal/quantization_util.cc',
'lib/tflm-cmsis/tensorflow/lite/kernels/internal/reference/comparisons.cc',
'lib/tflm-cmsis/tensorflow/lite/kernels/internal/reference/portable_tensor_utils.cc',
'lib/tflm-cmsis/tensorflow/lite/kernels/internal/runtime_shape.cc',
'lib/tflm-cmsis/tensorflow/lite/kernels/internal/tensor_ctypes.cc',
'lib/tflm-cmsis/tensorflow/lite/kernels/internal/tensor_utils.cc',
'lib/tflm-cmsis/tensorflow/lite/kernels/kernel_util.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/arena_allocator/non_persistent_arena_buffer_allocator.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/arena_allocator/persistent_arena_buffer_allocator.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/arena_allocator/recording_single_arena_buffer_allocator.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/arena_allocator/single_arena_buffer_allocator.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/cortex_m_generic/debug_log.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/cortex_m_generic/micro_time.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/fake_micro_context.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/flatbuffer_utils.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/hexdump.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/activations.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/activations_common.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/add_common.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/add_n.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/arg_min_max.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/assign_variable.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/batch_matmul_common.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/batch_to_space_nd.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/broadcast_args.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/broadcast_to.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/call_once.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/cast.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/ceil.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/circular_buffer.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/circular_buffer_common.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/cmsis_nn/add.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/cmsis_nn/batch_matmul.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/cmsis_nn/conv.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/cmsis_nn/depthwise_conv.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/cmsis_nn/fully_connected.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/cmsis_nn/maximum_minimum.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/cmsis_nn/mul.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/cmsis_nn/pooling.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/cmsis_nn/softmax.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/cmsis_nn/svdf.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/cmsis_nn/transpose_conv.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/cmsis_nn/unidirectional_sequence_lstm.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/comparisons.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/concatenation.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/conv_common.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/cumsum.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/decompress.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/decompress_common.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/depth_to_space.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/depthwise_conv_common.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/dequantize.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/dequantize_common.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/detection_postprocess.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/div.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/elementwise.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/elu.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/embedding_lookup.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/ethosu.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/expand_dims.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/exp.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/fill.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/floor.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/floor_div.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/floor_mod.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/fully_connected_common.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/gather.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/gather_nd.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/hard_swish.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/hard_swish_common.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/if.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/kernel_runner.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/kernel_util.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/l2norm.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/l2_pool_2d.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/leaky_relu.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/leaky_relu_common.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/logical.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/logical_common.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/logistic.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/logistic_common.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/log_softmax.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/lstm_eval.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/lstm_eval_common.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/micro_tensor_utils.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/mirror_pad.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/mul_common.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/neg.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/pack.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/pad.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/pooling_common.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/prelu.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/prelu_common.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/quantize.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/quantize_common.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/read_variable.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/reduce.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/reduce_common.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/reshape.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/reshape_common.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/resize_bilinear.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/resize_nearest_neighbor.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/round.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/select.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/shape.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/slice.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/softmax_common.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/space_to_batch_nd.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/space_to_depth.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/split.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/split_v.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/squared_difference.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/squeeze.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/strided_slice.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/strided_slice_common.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/sub.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/sub_common.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/svdf_common.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/tanh.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/transpose.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/unpack.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/var_handle.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/while.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/kernels/zeros_like.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/memory_helpers.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/memory_planner/greedy_memory_planner.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/memory_planner/linear_memory_planner.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/memory_planner/non_persistent_buffer_planner_shim.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/micro_allocation_info.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/micro_context.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/micro_interpreter.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/micro_interpreter_context.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/micro_log.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/micro_op_resolver.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/micro_profiler.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/micro_resource_variable.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/models/person_detect_model_data.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/system_setup.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/test_helper_custom_ops.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/test_helpers.cc',
'lib/tflm-cmsis/tensorflow/lite/micro/tflite_bridge/flatbuffer_conversions_bridge.cc',
)

add_project_arguments('-DCPU_MCXN947VDF_cm33_core0', language: ['c', 'cpp'])

mcux_sdk_inc = include_directories(
    'lib/mcux-sdk/devices',
    'lib/mcux-sdk/boards/frdmmcxn947',
    'lib/mcux-sdk/drivers/gpio',
    'lib/mcux-sdk/drivers/common',
    'lib/mcux-sdk/devices/MCXN947',
    'lib/mcux-sdk/devices/MCXN947/drivers',
    'lib/mcux-sdk/drivers/port'

)

# Source files for FreeRTOS
freertos_src = files(
    'lib/freertos/tasks.c',
    'lib/freertos/queue.c',
    'lib/freertos/timers.c',
    'lib/freertos/list.c',
    'lib/freertos/event_groups.c',
    'lib/freertos/portable/GCC/ARM_CM33/non_secure/port.c',
    'lib/freertos/portable/MemMang/heap_4.c',
    'lib/freertos/portable/GCC/ARM_CM33_NTZ/non_secure/portasm.c',
    'lib/freertos/portable/GCC/ARM_CM33_NTZ/non_secure/mpu_wrappers_v2_asm.c'
)

# Source files for CMSIS
cmsis_src = files(
)

# Source files for MCUX-SDK
mcux_sdk_src = files(
    'lib/mcux-sdk/devices/MCXN947/system_MCXN947_cm33_core0.c',
    'lib/mcux-sdk/devices/MCXN947/gcc/startup_MCXN947_cm33_core0.S',
    'lib/mcux-sdk/drivers/gpio/fsl_gpio.c',
    'lib/mcux-sdk/drivers/common/fsl_common.c'
)

# Application source
main_src = files(
    'src/main.c',
    'src/blinky.c',
    'src/inference.cpp',
    'src/model.c',
)

# Define the executable
executable(
    'freertos_project.elf',
    sources : [main_src, freertos_src, mcux_sdk_src, tflite_src],
    include_directories : [freertos_inc, mcux_sdk_inc, tflite_inc, cmsis_inc],
    c_args : c_args,
    cpp_args : cpp_args,
    link_args : ld_args
)


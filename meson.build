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
cpp_args = ['-lnosys', '--specs=nosys.specs', '-std=c++17', '-Os', '-ffunction-sections', '-fdata-sections', '-fno-rtti', '-Wno-unused-parameter', '-fno-exceptions']
ld_args = [
  '-T', '../config/MCXN947_cm33_core0_flash.ld',
  '-mcpu=cortex-m33', '-mthumb', '-mfloat-abi=soft',
  '-lc', '-lstdc++', '-lnosys', '--specs=nosys.specs'
]

add_project_arguments(
'-DCMSIS_NN',
'-DTF_LITE_DISABLE_DEFAULT_REPORT_ERRORS',
'-DTF_LITE_STATIC_MEMORY',
'-DCMSIS_DEVICE_ARM_CORTEX_M_XX_HEADER_FILE="ARMCM33.h"',
'-DTF_LITE_DISABLE_X86_NEON',
'-DCPU_MCXN947VDF_cm33_core0',
'-DSERIAL_PORT_TYPE_UART',
language: ['cpp','c'],
)

# '-DTF_LITE_STRIP_ERROR_STRINGS',

tflite_inc = include_directories(
    'lib/tflm-cmsis/',
    'lib/tflm-cmsis/third_party/flatbuffers/include',
    'lib/tflm-cmsis/third_party/gemmlowp',
    'lib/tflm-cmsis/third_party/ruy',
    'lib/tflm-cmsis/third_party/kissfft',
    'model',
    'lib/tflm-cmsis/third_party/cmsis_nn',
    'lib/tflm-cmsis/third_party/cmsis_nn/Include/',
    'lib/tflm-cmsis/tensorflow/lite/micro/cortex_m_generic',
)

tflite_src = files(
  'lib/tflm-cmsis/tensorflow/compiler/mlir/lite/core/api/error_reporter.cc',
  'lib/tflm-cmsis/tensorflow/compiler/mlir/lite/schema/schema_utils.cc',
  'lib/tflm-cmsis/tensorflow/lite/core/api/flatbuffer_conversions.cc',
  'lib/tflm-cmsis/tensorflow/lite/core/c/common.cc',
  'lib/tflm-cmsis/tensorflow/lite/kernels/internal/quantization_util.cc',
  'lib/tflm-cmsis/tensorflow/lite/kernels/internal/tensor_ctypes.cc',
  'lib/tflm-cmsis/tensorflow/lite/kernels/internal/common.cc',
  'lib/tflm-cmsis/tensorflow/lite/micro/micro_utils.cc',
  'lib/tflm-cmsis/tensorflow/lite/kernels/kernel_util.cc',
  'lib/tflm-cmsis/tensorflow/lite/micro/arena_allocator/single_arena_buffer_allocator.cc',
  'lib/tflm-cmsis/tensorflow/lite/micro/flatbuffer_utils.cc',
  'lib/tflm-cmsis/tensorflow/lite/micro/kernels/activations_common.cc',
  'lib/tflm-cmsis/third_party/cmsis_nn/Source/BasicMathFunctions/arm_elementwise_add_s8.c',
  'lib/tflm-cmsis/third_party/cmsis_nn/Source/BasicMathFunctions/arm_elementwise_add_s16.c',
  'lib/tflm-cmsis/tensorflow/lite/micro/kernels/reduce_common.cc',
  'lib/tflm-cmsis/tensorflow/lite/micro/kernels/activations.cc',
  'lib/tflm-cmsis/tensorflow/lite/micro/kernels/arg_min_max.cc',
  'lib/tflm-cmsis/tensorflow/lite/micro/kernels/pad.cc',
  'lib/tflm-cmsis/tensorflow/lite/micro/kernels/cmsis_nn/add.cc',
  'lib/tflm-cmsis/tensorflow/lite/micro/kernels/reduce.cc',
  'lib/tflm-cmsis/tensorflow/lite/micro/kernels/cmsis_nn/conv.cc',
  'lib/tflm-cmsis/tensorflow/lite/micro/kernels/cmsis_nn/depthwise_conv.cc',
  'lib/tflm-cmsis/tensorflow/lite/micro/kernels/cmsis_nn/fully_connected.cc',
  'lib/tflm-cmsis/tensorflow/lite/micro/kernels/cmsis_nn/softmax.cc',
  'lib/tflm-cmsis/tensorflow/lite/micro/kernels/conv_common.cc',
  'lib/tflm-cmsis/tensorflow/lite/micro/kernels/depthwise_conv_common.cc',
  'lib/tflm-cmsis/tensorflow/lite/micro/kernels/depthwise_conv_common.cc',
  'lib/tflm-cmsis/tensorflow/lite/micro/kernels/div.cc',
  'lib/tflm-cmsis/tensorflow/lite/micro/kernels/fully_connected_common.cc',
  'lib/tflm-cmsis/tensorflow/lite/micro/kernels/kernel_util.cc',
  'lib/tflm-cmsis/tensorflow/lite/micro/kernels/kernel_util.cc',
  'lib/tflm-cmsis/tensorflow/lite/micro/kernels/softmax_common.cc',
  'lib/tflm-cmsis/tensorflow/lite/micro/memory_helpers.cc',
  'lib/tflm-cmsis/tensorflow/lite/micro/memory_helpers.cc',
  'lib/tflm-cmsis/tensorflow/lite/micro/memory_planner/greedy_memory_planner.cc',
  'lib/tflm-cmsis/tensorflow/lite/micro/memory_planner/linear_memory_planner.cc',
  'lib/tflm-cmsis/tensorflow/lite/micro/micro_allocation_info.cc',
  'lib/tflm-cmsis/tensorflow/lite/micro/micro_allocator.cc',
  'lib/tflm-cmsis/tensorflow/lite/micro/micro_context.cc',
  'lib/tflm-cmsis/tensorflow/lite/micro/micro_interpreter.cc',
  'lib/tflm-cmsis/tensorflow/lite/micro/micro_interpreter_context.cc',
  'lib/tflm-cmsis/tensorflow/lite/micro/micro_interpreter_graph.cc',
  'lib/tflm-cmsis/tensorflow/lite/micro/micro_log.cc',
  'lib/tflm-cmsis/tensorflow/lite/micro/micro_op_resolver.cc',
  'lib/tflm-cmsis/tensorflow/lite/micro/micro_resource_variable.cc',
  'lib/tflm-cmsis/tensorflow/lite/micro/tflite_bridge/flatbuffer_conversions_bridge.cc',
  'lib/tflm-cmsis/tensorflow/lite/micro/tflite_bridge/micro_error_reporter.cc',
  'lib/tflm-cmsis/third_party/cmsis_nn/Source/ConvolutionFunctions/arm_convolve_1_x_n_s4.c',
  'lib/tflm-cmsis/third_party/cmsis_nn/Source/ConvolutionFunctions/arm_convolve_1_x_n_s8.c',
  'lib/tflm-cmsis/third_party/cmsis_nn/Source/ConvolutionFunctions/arm_convolve_1x1_s4.c',
  'lib/tflm-cmsis/third_party/cmsis_nn/Source/ConvolutionFunctions/arm_convolve_1x1_s4_fast.c',
  'lib/tflm-cmsis/third_party/cmsis_nn/Source/ConvolutionFunctions/arm_convolve_1x1_s8.c',
  'lib/tflm-cmsis/third_party/cmsis_nn/Source/ConvolutionFunctions/arm_convolve_1x1_s8_fast.c',
  'lib/tflm-cmsis/third_party/cmsis_nn/Source/ConvolutionFunctions/arm_convolve_get_buffer_sizes_s16.c',
  'lib/tflm-cmsis/third_party/cmsis_nn/Source/ConvolutionFunctions/arm_convolve_get_buffer_sizes_s4.c',
  'lib/tflm-cmsis/third_party/cmsis_nn/Source/ConvolutionFunctions/arm_convolve_get_buffer_sizes_s8.c',
  'lib/tflm-cmsis/third_party/cmsis_nn/Source/ConvolutionFunctions/arm_convolve_get_buffer_sizes_s8.c',
  'lib/tflm-cmsis/third_party/cmsis_nn/Source/ConvolutionFunctions/arm_convolve_s16.c',
  'lib/tflm-cmsis/third_party/cmsis_nn/Source/ConvolutionFunctions/arm_convolve_s4.c',
  'lib/tflm-cmsis/third_party/cmsis_nn/Source/ConvolutionFunctions/arm_convolve_s8.c',
  'lib/tflm-cmsis/third_party/cmsis_nn/Source/ConvolutionFunctions/arm_convolve_wrapper_s16.c',
  'lib/tflm-cmsis/third_party/cmsis_nn/Source/ConvolutionFunctions/arm_convolve_wrapper_s4.c',
  'lib/tflm-cmsis/third_party/cmsis_nn/Source/ConvolutionFunctions/arm_convolve_wrapper_s8.c',
  'lib/tflm-cmsis/third_party/cmsis_nn/Source/ConvolutionFunctions/arm_depthwise_conv_3x3_s8.c',
  'lib/tflm-cmsis/third_party/cmsis_nn/Source/ConvolutionFunctions/arm_depthwise_conv_get_buffer_sizes_s4.c',
  'lib/tflm-cmsis/third_party/cmsis_nn/Source/ConvolutionFunctions/arm_depthwise_conv_get_buffer_sizes_s8.c',
  'lib/tflm-cmsis/third_party/cmsis_nn/Source/ConvolutionFunctions/arm_depthwise_conv_s16.c',
  'lib/tflm-cmsis/third_party/cmsis_nn/Source/ConvolutionFunctions/arm_depthwise_conv_s4.c',
  'lib/tflm-cmsis/third_party/cmsis_nn/Source/ConvolutionFunctions/arm_depthwise_conv_s4_opt.c',
  'lib/tflm-cmsis/third_party/cmsis_nn/Source/ConvolutionFunctions/arm_depthwise_conv_s8.c',
  'lib/tflm-cmsis/third_party/cmsis_nn/Source/ConvolutionFunctions/arm_depthwise_conv_s8_opt.c',
  'lib/tflm-cmsis/third_party/cmsis_nn/Source/ConvolutionFunctions/arm_depthwise_conv_wrapper_s4.c',
  'lib/tflm-cmsis/third_party/cmsis_nn/Source/ConvolutionFunctions/arm_depthwise_conv_wrapper_s8.c',
  'lib/tflm-cmsis/third_party/cmsis_nn/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_row_offset_s8_s16.c',
  'lib/tflm-cmsis/third_party/cmsis_nn/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s16.c',
  'lib/tflm-cmsis/third_party/cmsis_nn/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s4_s16.c',
  'lib/tflm-cmsis/third_party/cmsis_nn/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s8_s16.c',
  'lib/tflm-cmsis/third_party/cmsis_nn/Source/FullyConnectedFunctions/arm_fully_connected_get_buffer_sizes_s16.c',
  'lib/tflm-cmsis/third_party/cmsis_nn/Source/FullyConnectedFunctions/arm_fully_connected_get_buffer_sizes_s8.c',
  'lib/tflm-cmsis/third_party/cmsis_nn/Source/FullyConnectedFunctions/arm_fully_connected_per_channel_s8.c',
  'lib/tflm-cmsis/third_party/cmsis_nn/Source/FullyConnectedFunctions/arm_fully_connected_s16.c',
  'lib/tflm-cmsis/third_party/cmsis_nn/Source/FullyConnectedFunctions/arm_fully_connected_s4.c',
  'lib/tflm-cmsis/third_party/cmsis_nn/Source/FullyConnectedFunctions/arm_fully_connected_s8.c',
  'lib/tflm-cmsis/third_party/cmsis_nn/Source/FullyConnectedFunctions/arm_fully_connected_wrapper_s8.c',
  'lib/tflm-cmsis/third_party/cmsis_nn/Source/FullyConnectedFunctions/arm_vector_sum_s8.c',
  'lib/tflm-cmsis/third_party/cmsis_nn/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s4.c',
  'lib/tflm-cmsis/third_party/cmsis_nn/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s8.c',
  'lib/tflm-cmsis/third_party/cmsis_nn/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_per_ch_s8.c',
  'lib/tflm-cmsis/third_party/cmsis_nn/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s16.c',
  'lib/tflm-cmsis/third_party/cmsis_nn/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s4.c',
  'lib/tflm-cmsis/third_party/cmsis_nn/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s8.c',
  'lib/tflm-cmsis/third_party/cmsis_nn/Source/NNSupportFunctions/arm_q7_to_q15_with_offset.c',
  'lib/tflm-cmsis/third_party/cmsis_nn/Source/SoftmaxFunctions/arm_nn_softmax_common_s8.c',
  'lib/tflm-cmsis/third_party/cmsis_nn/Source/SoftmaxFunctions/arm_softmax_s16.c',
  'lib/tflm-cmsis/third_party/cmsis_nn/Source/SoftmaxFunctions/arm_softmax_s8.c',
  'lib/tflm-cmsis/third_party/cmsis_nn/Source/SoftmaxFunctions/arm_softmax_s8_s16.c',
  'lib/tflm-cmsis/tensorflow/lite/micro/cortex_m_generic/debug_log.cc',
  'lib/tflm-cmsis/tensorflow/lite/micro/kernels/quantize.cc',
  'lib/tflm-cmsis/tensorflow/lite/micro/kernels/quantize_common.cc',
  'lib/tflm-cmsis/tensorflow/lite/micro/kernels/logistic.cc',
  'lib/tflm-cmsis/tensorflow/lite/micro/kernels/logistic_common.cc',
)

cmsis_inc = include_directories(
    'lib/cmsis/CMSIS/Core/Include',
    'lib/cmsis/Device/ARM/ARMCM33/Include/'
)

# Source files for CMSIS
cmsis_src = files(
)

# Include directories
freertos_inc = include_directories(
    'lib/freertos/include',
    'lib/freertos/portable/GCC/ARM_CM33/non_secure/',
    'lib/freertos/portable/GCC/ARM_CM33/secure',
    'config',
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

mcux_sdk_inc = include_directories(
    'lib/mcux-sdk/devices',
    'lib/mcux-sdk/boards/frdmmcxn947',
    'lib/mcux-sdk/drivers/gpio',
    'lib/mcux-sdk/drivers/uart',
    'lib/mcux-sdk/drivers/common',
    'lib/mcux-sdk/devices/MCXN947',
    'lib/mcux-sdk/devices/MCXN947/drivers',
    'lib/mcux-sdk/drivers/port',
    'lib/mcux-sdk/drivers/flexspi',
    'lib/mcux-sdk/drivers/lpflexcomm/lpuart',
    'lib/mcux-sdk/drivers/lpflexcomm/',
    'lib/mcux-sdk/drivers/mcx_spc/',
)

# Source files for MCUX-SDK
mcux_sdk_src = files(
    'lib/mcux-sdk/devices/MCXN947/drivers/fsl_clock.c',
    'lib/mcux-sdk/devices/MCXN947/drivers/fsl_reset.c',
    'lib/mcux-sdk/devices/MCXN947/gcc/startup_MCXN947_cm33_core0.S',
    'lib/mcux-sdk/devices/MCXN947/system_MCXN947_cm33_core0.c',
    'lib/mcux-sdk/drivers/flexspi/fsl_flexspi.c',
    'lib/mcux-sdk/drivers/gpio/fsl_gpio.c',
    'lib/mcux-sdk/drivers/mcx_spc/fsl_spc.c',
    'lib/mcux-sdk/drivers/lpflexcomm/lpuart/fsl_lpuart.c',
    'lib/mcux-sdk/drivers/lpflexcomm/lpuart/fsl_lpuart_freertos.c',
    'lib/mcux-sdk/drivers/lpflexcomm/fsl_lpflexcomm.c',
)

# Application source
main_inc = include_directories(
  'src'
)

main_src = files(
    'src/blinky.c',
    'src/clock_config.c',
    'src/flexspi.c',
    'src/inference.cpp',
    'src/model.c',
    'src/pin_mux.c',
    'src/uart.c',
    'src/main.c',
)

# Define the executable
executable(
    'freertos_project.elf',
    sources : [main_src, freertos_src, mcux_sdk_src, tflite_src],
    include_directories : [main_inc, freertos_inc, mcux_sdk_inc, tflite_inc, cmsis_inc],
    c_args : c_args,
    cpp_args : cpp_args,
    link_args : ld_args
)


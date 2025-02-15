project(
    'MCX hello world',
    ['c','cpp'],
    version : '0.1',
    default_options : ['warning_level=2', 'c_std=c11', 'cpp_std=c++11']
)

# Define target architecture and toolchain
toolchain = 'arm-none-eabi-'
cc = meson.get_compiler('c')
cpp = meson.get_compiler('cpp')

# Common CPU arguments
cpu_args = [
    '-nostdlib',
    '-lnosys'
]

# Apply CPU arguments to both C and C++
c_args = cpu_args
cpp_args = cpu_args

ld_args = [
    '-T', '../lib/mcux-sdk/devices/MCXN947/gcc/MCXN947_cm33_core0_flash_s.ld',
    '-lc',   # Link C standard library
    '-lm',   # Link standard math library
    '-lstdc++',  # Ensure C++ standard library is linked
    '-lsupc++',  # Link C++ runtime support library
    '-lnosys',
    '--specs=nano.specs'
]

# Include directories
freertos_inc = include_directories(
    'lib/freertos/include',
    'lib/freertos/portable/GCC/ARM_CM33/non_secure/',
    'lib/freertos/portable/GCC/ARM_CM33/secure',
    'config',
)
cmsis_inc = include_directories('lib/cmsis/CMSIS/Core/Include/')

add_project_arguments('-DCPU_MCXN947VDF_cm33_core0', language: 'c')

tensorflow_inc = include_directories(
  'lib/tflite-micro',
  'lib/flatbuffers/include'
  )

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
    'lib/freertos/portable/MemMang/heap_4.c',
    'lib/freertos/portable/GCC/ARM_CM33/non_secure/port.c',  # Remove TrustZone NTZ files
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
    'src/model.cpp'
)

main_inc = include_directories(
    'src/',
    'model/'
)

# Define the executable
executable(
    'freertos_project.elf',
    sources : [main_src, freertos_src, cmsis_src, mcux_sdk_src],
    include_directories : [main_inc, freertos_inc, cmsis_inc, mcux_sdk_inc, tensorflow_inc],
    c_args : c_args,
    cpp_args : cpp_args,
    link_args : ld_args
)


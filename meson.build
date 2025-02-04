project(
    'MCX hello world',
    'c',
    version : '0.1',
    default_options : ['warning_level=2', 'c_std=c11']
)

# Define target architecture and toolchain
toolchain = 'arm-none-eabi-'
cc = meson.get_compiler('c')
c_args = ['-mcpu=cortex-m33', '-mthumb', '-nostdlib', '-lnosys']
ld_args = [
  '-T', '../lib/mcux-sdk/devices/MCXN947/gcc/MCXN947_cm33_core0_flash_s.ld',
  '-lc',               # Link the C standard library
  '-lnosys',           # Provide minimal stubs
  '--specs=nano.specs', # Use nano version of Newlib
] # Replace with actual linker script

# Include directories
freertos_inc = include_directories(
    'lib/freertos/include',
    'lib/freertos/portable/GCC/ARM_CM33/non_secure/',
    'lib/freertos/portable/GCC/ARM_CM33/secure',
    'config',
)
cmsis_inc = include_directories('lib/cmsis/CMSIS/Core/Include/')

add_project_arguments('-DCPU_MCXN947VDF_cm33_core0', language: 'c')

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
    'src/blinky.c'
)

# Define the executable
executable(
    'freertos_project.elf',
    sources : [main_src, freertos_src, cmsis_src, mcux_sdk_src],
    include_directories : [freertos_inc, cmsis_inc, mcux_sdk_inc],
    c_args : c_args,
    link_args : ld_args
)


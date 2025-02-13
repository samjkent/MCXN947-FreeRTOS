ELF_FILE := "build/freertos_project.elf"

setup:
  docker build -t arm-gcc-env .

build:
  docker run --volume "$(pwd)":"$(pwd)" --workdir "$(pwd)" arm-gcc-env

flash: build
  /usr/local/LinkServer_24.12.21/LinkServer flash --probe 0Q5BWDEVI42LX MCXN947:FRDM-MCXN947 load {{ELF_FILE}}

debug: build
  ./scripts/debug.sh {{ELF_FILE}}

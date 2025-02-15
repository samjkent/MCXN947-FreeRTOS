ELF_FILE := "build/freertos_project.elf"
COMPILE_COMMANDS := "build/compile_commands.json"

setup:
  docker build -t arm-gcc-env .

build:
  docker run --user $(id -u):$(id -g) --volume "$(pwd)":"$(pwd)" --workdir "$(pwd)" arm-gcc-env

alias b := build

flash: build
  /usr/local/LinkServer_24.12.21/LinkServer flash --probe 0Q5BWDEVI42LX MCXN947:FRDM-MCXN947 load {{ELF_FILE}}

debug: build
  ./scripts/debug.sh {{ELF_FILE}}

clean:
  rm -rf build

docker build -t arm-gcc-env .
docker run --volume "$(pwd)":"$(pwd)" --workdir "$(pwd)" arm-gcc-env

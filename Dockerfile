# Use a lightweight base image
FROM ubuntu:22.04

# Set environment variables for non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive

# Install required dependencies
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    ninja-build \
    gdb-multiarch \
    libncurses-dev \
    libnewlib-arm-none-eabi \
    picolibc-arm-none-eabi \
    gcc-arm-none-eabi \
    meson \
    && rm -rf /var/lib/apt/lists/*

# Add ARM GCC to PATH
ENV PATH="/usr/local/gcc-arm-none-eabi-12.2.rel1/bin:$PATH"

# Verify installation
RUN arm-none-eabi-gcc --version

# Set up workspace
WORKDIR /workspace

# Default command: Open a shell
CMD ["sh", "./scripts/build-cross.sh"]

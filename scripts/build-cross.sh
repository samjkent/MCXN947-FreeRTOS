if [ ! -d "build" ]; then
  echo "Setting up build"
  meson setup build --cross-file=arm-gcc.txt
fi
cd build && ninja
chmod 775 compile_commands.json

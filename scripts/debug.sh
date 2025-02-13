#!/bin/bash

# Start LinkServer in the background
/usr/local/LinkServer_24.12.21/LinkServer gdbserver --probe 0Q5BWDEVI42LX MCXN947:FRDM-MCXN947 &
# Capture the process ID of LinkServer
LINKSERVER_PID=$!

# Define a cleanup function to stop LinkServer when the script exits
cleanup() {
  echo "Stopping LinkServer (PID: $LINKSERVER_PID)..."
  kill $LINKSERVER_PID 2>/dev/null
}

# Trap EXIT signal to run cleanup function when the script exits
trap cleanup EXIT

# Wait for a short period to ensure LinkServer is up
sleep 2

# Start gdb-multiarch with the .gdbinit config
gdb-multiarch -x .gdbinit $1

# After gdb exits, cleanup function will run automatically

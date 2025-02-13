# Clear all existing breakpoints
delete breakpoints

# Enable TUI mode and switch to source+assembly view
set pagination off
layout src

# Connect to the remote target
target remote :3333

# Load file and symbols from the ELF binary
load

# Set a breakpoint at main
break main

# Continue execution
continue

# Redraw to fix the TUI
refresh

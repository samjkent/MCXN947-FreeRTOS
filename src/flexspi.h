#ifndef FLEXSPI_DRIVER_H
#define FLEXSPI_DRIVER_H

#include "fsl_flexspi.h"

// Define the base address of the FlexSPI peripheral
#define FLEXSPI_BASE FLEXSPI0

// Function to initialize the FlexSPI interface
void flexspi_init(void);

void flexspi_enable_memory_map(void);

#endif // FLEXSPI_DRIVER_H

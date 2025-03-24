#include "flexspi.h"
#include "fsl_clock.h"
#include "fsl_reset.h"
#include "fsl_port.h"

#define CUSTOM_LUT_LENGTH 643

#define CMD_INDEX_QUAD_READ       0  // Index for Quad Read
#define CMD_INDEX_PAGE_PROGRAM    1  // Index for Page Program
#define CMD_INDEX_SECTOR_ERASE    2  // Index for Sector Erase
#define CMD_INDEX_READ_STATUS     3  // Index for Read Status Register

const uint32_t flexspi_lut[] = {
    // Quad Input Fast Read (0xEB)
    [4 * CMD_INDEX_QUAD_READ] =
        FLEXSPI_LUT_SEQ(
            kFLEXSPI_Command_SDR, kFLEXSPI_1PAD, 0xEB,
            kFLEXSPI_Command_RADDR_SDR, kFLEXSPI_4PAD, 0x18),
    [4 * CMD_INDEX_QUAD_READ + 1] =
        FLEXSPI_LUT_SEQ(
            kFLEXSPI_Command_DUMMY_SDR, kFLEXSPI_4PAD, 0x06,
            kFLEXSPI_Command_READ_SDR, kFLEXSPI_4PAD, 0x04),

    // Page Program (0x02)
    [4 * CMD_INDEX_PAGE_PROGRAM] =
        FLEXSPI_LUT_SEQ(
            kFLEXSPI_Command_SDR, kFLEXSPI_1PAD, 0x02,
            kFLEXSPI_Command_RADDR_SDR, kFLEXSPI_1PAD, 0x18),
    [4 * CMD_INDEX_PAGE_PROGRAM + 1] =
        FLEXSPI_LUT_SEQ(
            kFLEXSPI_Command_WRITE_SDR, kFLEXSPI_1PAD, 0x04,
            kFLEXSPI_Command_STOP, kFLEXSPI_1PAD, 0x00),

    // Sector Erase (4KB) (0x20)
    [4 * CMD_INDEX_SECTOR_ERASE] =
        FLEXSPI_LUT_SEQ(
            kFLEXSPI_Command_SDR, kFLEXSPI_1PAD, 0x20,
            kFLEXSPI_Command_RADDR_SDR, kFLEXSPI_1PAD, 0x18),

    // Read Status Register-1 (0x05)
    [4 * CMD_INDEX_READ_STATUS] =
        FLEXSPI_LUT_SEQ(
            kFLEXSPI_Command_SDR, kFLEXSPI_1PAD, 0x05,
            kFLEXSPI_Command_READ_SDR, kFLEXSPI_1PAD, 0x04),
};

void flexspi_pin_config() {
    CLOCK_EnableClock(kCLOCK_Port3);
    PORT_SetPinMux(PORT3, 0, kPORT_MuxAlt8); // FLEXSPI0_A_SS0_b
    PORT_SetPinMux(PORT3, 7, kPORT_MuxAlt8); // FLEXSPI0_A_SCLK
    PORT_SetPinMux(PORT3, 8, kPORT_MuxAlt8); // FLEXSPI0_A_DATA0
    PORT_SetPinMux(PORT3, 9, kPORT_MuxAlt8); // FLEXSPI0_A_DATA1
    PORT_SetPinMux(PORT3, 10, kPORT_MuxAlt8); // FLEXSPI0_A_DATA2
    PORT_SetPinMux(PORT3, 11, kPORT_MuxAlt8); // FLEXSPI0_A_DATA3

}

void flexspi_init(void) {

    flexspi_pin_config();

    flexspi_config_t config;

    CLOCK_SetClkDiv(kCLOCK_DivFlexspiClk, 2U);
    CLOCK_AttachClk(kFRO_HF_to_FLEXSPI);

    // Get default config
    FLEXSPI_GetDefaultConfig(&config);
    // Set device specific config
    config.ahbConfig.enableAHBPrefetch = true;
    config.ahbConfig.enableAHBBufferable = true;
    config.ahbConfig.enableReadAddressOpt = true;
    config.ahbConfig.enableAHBCachable = true;
    config.rxSampleClock = kFLEXSPI_ReadSampleClkLoopbackFromDqsPad;

    flexspi_device_config_t device_config = {
      .flexspiRootClk = 48000000,                             // 48 MHz FlexSPI clock
      .isSck2Enabled = false,                                 // Not mentioned, assuming default false
      .flashSize = 0x2000,                                    // 8MB = 8192KB = 0x2000
      .CSIntervalUnit = kFLEXSPI_CsIntervalUnit1SckCycle,     // Interval unit in SCK cycles
      .CSInterval = 2,                                        // Minimum interval of 2 SCK cycles
      .CSHoldTime = 3,                                        // CS hold time >= 9.8ns @75MHz (~13.3ns = 1 cycle) → 3 cycles
      .CSSetupTime = 3,                                       // CS setup time ≥ 8.3ns → 3 cycles
      .dataValidTime = 2,                                     // Data valid delay in nanoseconds (used in DLLACR/BCR)
      .columnspace = 3,                                       // 3-bit column address width
      .enableWordAddress = true,                              // Enable 2-byte word addressing (16-bit access)
      .AWRSeqIndex = 1,                                       // LUT index for write sequence
      .AWRSeqNumber = 1,                                      // 1 write sequence
      .ARDSeqIndex = 0,                                       // LUT index for read sequence
      .ARDSeqNumber = 1,                                      // 1 read sequence
      .AHBWriteWaitUnit = kFLEXSPI_AhbWriteWaitUnit2AhbCycle, // Default setting if not specified
      .AHBWriteWaitInterval = 0,                              // Default wait interval
      .enableWriteMask = true                                 // Enable DQS write mask for 16-bit alignment
    };

    // Initialize FlexSPI with the configuration
    FLEXSPI_Init(FLEXSPI0, &config);

    /* Configure flash settings according to serial flash feature. */
    FLEXSPI_SetFlashConfig(FLEXSPI0, &device_config, kFLEXSPI_PortA1);

    /* Update LUT table. */
    FLEXSPI_UpdateLUT(FLEXSPI0, 0, flexspi_lut, sizeof(flexspi_lut));

    /* Do software reset. */
    FLEXSPI_SoftwareReset(FLEXSPI0);
    FLEXSPI_Enable(FLEXSPI0, true);
}


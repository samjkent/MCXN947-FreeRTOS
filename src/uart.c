#include "uart.h"
#include "MCXN947_cm33_core0.h"
#include <stdarg.h>
#include "clock_config.h"

#define UART_PRINTF_BUFFER_SIZE 128

#define ANSI_COLOR_RED     "\033[31m"
#define ANSI_COLOR_YELLOW  "\033[33m"
#define ANSI_COLOR_RESET   "\033[0m"

#define LOG_NONE  0
#define LOG_ERROR 1
#define LOG_WARN  2
#define LOG_INFO  3

#define LOG_LEVEL LOG_INFO

lpuart_rtos_handle_t lpuart_rtos_handle;
lpuart_handle_t lpuart_handle;
static uint8_t lpuart_rx_buffer[256];  // Size as needed

void uart_init() {
  CLOCK_AttachClk(kFRO_HF_DIV_to_FLEXCOMM4);
  CLOCK_SetClkDiv(kCLOCK_DivFlexcom4Clk, 2U);

  lpuart_rtos_config_t config = {
    .base = LPUART4,
    .baudrate = 115200,
    .parity = kLPUART_ParityDisabled,
    .stopbits = kLPUART_OneStopBit,
    .buffer = lpuart_rx_buffer,
    .buffer_size = sizeof(lpuart_rx_buffer),
  };
  config.srcclk = CLOCK_GetLPFlexCommClkFreq(4);

  NVIC_SetPriority(LP_FLEXCOMM4_IRQn, configMAX_SYSCALL_INTERRUPT_PRIORITY - 1);
  assert(LPUART_RTOS_Init(&lpuart_rtos_handle, &lpuart_handle, &config) == kStatus_Success);
  LPUART_RTOS_SetTxTimeout(&lpuart_rtos_handle, 100, 1);
  LPUART_RTOS_SetRxTimeout(&lpuart_rtos_handle, 100, 1);
}

int uart_printf(const char *format, ...)
{
    char buffer[UART_PRINTF_BUFFER_SIZE];
    buffer[0] = 0;

    va_list args;
    va_start(args, format);

    // Format the string into the buffer
    int len = vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);

    // Ensure null-termination
    if (len < 0 || len >= sizeof(buffer)) {
        len = sizeof(buffer) - 1;
        buffer[len] = '\0';
    }

    // Send the formatted string over UART
    if(lpuart_rtos_handle.txSemaphore != NULL) {
      return LPUART_RTOS_Send(&lpuart_rtos_handle, (uint8_t *)buffer, len);
    }
}

void logi(const char *format, ...)
{
    va_list args;
    va_start(args, format);

  #if LOG_LEVEL >= LOG_INFO
    uart_printf(format, args);
    uart_printf("\r\n");
  #endif


    va_end(args);
}

void logw(const char *format, ...)
{
    va_list args;
    va_start(args, format);

  #if LOG_LEVEL >= LOG_WARN
    uart_printf(ANSI_COLOR_YELLOW);
    uart_printf(format, args);
    uart_printf("\r\n");
    uart_printf(ANSI_COLOR_RESET);
  #endif

    va_end(args);
}

void loge(const char *format, ...)
{
    va_list args;
    va_start(args, format);

  #if LOG_LEVEL >= LOG_ERROR
    uart_printf(ANSI_COLOR_RED);
    uart_printf(format, args);
    uart_printf("\r\n");
    uart_printf(ANSI_COLOR_RESET);
  #endif

    va_end(args);
}

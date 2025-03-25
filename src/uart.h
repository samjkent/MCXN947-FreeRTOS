#ifndef UART_H
#define UART_H

#include "fsl_lpuart_freertos.h"
#include "fsl_lpuart.h"

void uart_init();
int uart_printf(const char *format, ...);

void logi(const char *format, ...);
void logw(const char *format, ...);
void loge(const char *format, ...);

#endif // UART_H

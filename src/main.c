#include "MCXN947_cm33_core0.h"
#include "FreeRTOS.h"
#include "task.h"
#include "blinky.h"
#include "model.h"
#include "clock_config.h"
#include "flexspi.h"
#include "stdio.h"
#include "pin_mux.h"
#include "uart.h"

void vApplicationStackOverflowHook(TaskHandle_t xTask, char *pcTaskName) {
    // Print the task name causing the overflow

    // Take appropriate actions (e.g., reset the system, log an error, etc.)
    for (;;);
}

void vTask2(void *pvParameters) {
    while (1) {
        // Task 2 code
        vTaskDelay(pdMS_TO_TICKS(500));
    }
}

void vTask1(void *pvParameters) {
  uart_printf("Test");
  while (1) {
    // Task 1 code
    vTaskDelay(pdMS_TO_TICKS(1000));

  }
}

int main(void) {
    
    BOARD_BootClockFROHF48M();
    BOARD_InitDEBUG_UARTPins();
    flexspi_init();
    uart_init();

    xTaskCreate(vInferenceTask, "Inference", 512, NULL, 3, NULL);
    xTaskCreate(vBlinkyTask, "Blinky", 256, NULL, 3, NULL);

    vTaskStartScheduler();

    while (1) {
        // Should never reach here
    }
}


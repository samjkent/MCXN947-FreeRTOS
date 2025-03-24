#include "MCXN947_cm33_core0.h"
#include "FreeRTOS.h"
#include "task.h"
#include "blinky.h"
#include "model.h"
#include "clock_config.h"
#include "flexspi.h"
#include "stdio.h"

void vApplicationStackOverflowHook(TaskHandle_t xTask, char *pcTaskName) {
    // Print the task name causing the overflow

    // Take appropriate actions (e.g., reset the system, log an error, etc.)
    for (;;);
}

void vTask1(void *pvParameters) {
    while (1) {
        // Task 1 code
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

void vTask2(void *pvParameters) {
    while (1) {
        // Task 2 code
        vTaskDelay(pdMS_TO_TICKS(500));
    }
}

int main(void) {

    BOARD_BootClockFROHF48M();
    BOARD_InitDEBUG_UARTPins();
    printf("hello, world!");

    flexspi_init();

    xTaskCreate(vBlinkyTask, "Blinky", 256, NULL, 2, NULL);
    xTaskCreate(vInferenceTask, "Inference", 2048, NULL, 2, NULL);

    vTaskStartScheduler();

    while (1) {
        // Should never reach here
    }
}


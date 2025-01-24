#include "FreeRTOS.h"
#include "task.h"

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
    xTaskCreate(vTask1, "Task1", 1000, NULL, 1, NULL);
    xTaskCreate(vTask2, "Task2", 1000, NULL, 1, NULL);

    vTaskStartScheduler();

    while (1) {
        // Should never reach here
    }
}


#include "model.h"
#include "FreeRTOS.h"
#include "task.h"
#include <stdio.h>
#include "inference.h"
#include "debug_log_callback.h"
#include "uart.h"

// Task handle
TaskHandle_t xInferTaskHandle = NULL;

void tflite_print(const char *string)
{
  return uart_printf(string);
}

// FreeRTOS task for inference
void vInferenceTask(void *pvParameters) {
  // Initialize the TFLite model

  logi("Registering debug_log_callback");
  RegisterDebugLogCallback(tflite_print);

  init_model();
 
  while (1) {
    // Wait for notification to start inference
    uint32_t ulNotificationValue;
    if (xTaskNotifyWait(0, 0, &ulNotificationValue, portMAX_DELAY) == pdTRUE) {
      // printf("Running inference on new image...\n");

      // Run the model inference
      // int result = run_inference();

      // Print the classification result
      // printf("Inference result: %d\n", result);
    }
  }
}

// Function to trigger inference from another task
void TriggerInference() {
  if (xInferTaskHandle != NULL) {
    xTaskNotify(xInferTaskHandle, 0, eNoAction);
  }
}

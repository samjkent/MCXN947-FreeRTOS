#include "model.h"
#include "FreeRTOS.h"
#include "task.h"
#include <stdio.h>

// Task handle
TaskHandle_t xInferTaskHandle = NULL;

// FreeRTOS task for inference
void InferenceTask(void *pvParameters) {
  // printf("Initializing TensorFlow Lite Model...\n");

  // Initialize the TFLite model
  // init_model();

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

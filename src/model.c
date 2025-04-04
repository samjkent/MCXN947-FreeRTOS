#include "model.h"
#include "FreeRTOS.h"
#include "debug_log_callback.h"
#include "inference.h"
#include "task.h"
#include "uart.h"
#include <stdio.h>

#include "images.h"

// Task handle
TaskHandle_t xInferTaskHandle = NULL;

void tflite_print(const char *string) { return uart_printf(string); }

void check_result(float result) {
  // Check if the result is greater than 0.5
  if (result > 0.5f) {
    logi("Cat detected");
  } else {
    logi("Dog detected");
  }
}

// FreeRTOS task for inference
void vInferenceTask(void *pvParameters) {
  float result = 0.0f;

  // Initialize the TFLite model
  logi("Registering debug_log_callback");
  RegisterDebugLogCallback(tflite_print);

  init_model();

  while (1) {
    // Wait for notification to start inference
    uint32_t ulNotificationValue;
    // if (xTaskNotifyWait(0, 0, &ulNotificationValue, portMAX_DELAY) == pdTRUE)
    // {
    logi("Running inference on new image");

    // Run the model inference
    result = run_inference(input_data_0, sizeof(input_data_0));
    logi("Image: Dog");
    check_result(result);

    result = run_inference(input_data_1, sizeof(input_data_1));
    logi("Image: Dog");
    check_result(result);

    result = run_inference(input_data_3, sizeof(input_data_1));
    logi("Image: Cat");
    check_result(result);

    // }
    vTaskDelay(pdMS_TO_TICKS(portMAX_DELAY));
  }
}

// Function to trigger inference from another task
void TriggerInference() {
  if (xInferTaskHandle != NULL) {
    xTaskNotify(xInferTaskHandle, 0, eNoAction);
  }
}

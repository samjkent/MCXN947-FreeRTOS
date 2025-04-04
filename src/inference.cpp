#include "inference.h"
#include "cat.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "uart.h"
#include <iostream>
#include <math.h> // for expf

constexpr int kTensorArenaSize = 200 * 1024; // 250KB (adjust as needed)
uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;

int init_model() {
  // Load model from memory
  model = tflite::GetModel((const void *)tflite_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model provided is schema version %d not equal to supported "
                "version %d.",
                model->version(), TFLITE_SCHEMA_VERSION);
    return 1;
  }

  // Declare the op resolver and register only the necessary operations
  static tflite::MicroMutableOpResolver<10>
      resolver; // Adjust NUM_OPS based on the model

  // Register required TensorFlow Lite operations used in your model
  resolver.AddAdd();
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddFullyConnected();
  resolver.AddHardSwish();
  resolver.AddLogistic();
  resolver.AddMean();
  resolver.AddMul();
  // resolver.AddPad();
  resolver.AddQuantize();
  resolver.AddSoftmax();

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return 1;
  }

  // Get information about the memory area to use for the model's input.
  input = interpreter->input(0);

  return 0;
}

float run_inference(void *ptr, uint32_t len) {
  if (len < (144 * 144 * 3)) {
    MicroPrintf("Invalid input size. Array length %lu", len);
    return 0.0f;
  }

  /* Load picture data to input tensor */
  for (int i = 0; i < 144 * 144 * 3; i++) {
    input->data.int8[i] = ((int8_t *)ptr)[i];
  }

  // Run the model on this input and make sure it succeeds.
  int rc = interpreter->Invoke();
  if (kTfLiteOk != rc) {
    MicroPrintf("Failed to invoke interpreter. rc = %i", rc);
    return 0.0f;
  }

  TfLiteTensor *output = interpreter->output(0);
  int8_t cat_score_q = output->data.int8[0];
  int8_t not_cat_score_q = output->data.int8[1];

  float scale = output->params.scale;
  int zero_point = output->params.zero_point;

  float cat_score = (cat_score_q - zero_point) * scale;
  float not_cat_score = (not_cat_score_q - zero_point) * scale;

  return cat_score;
}

#include "inference.h"
#include "cat.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
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
  model = tflite::GetModel(tflite_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    // loge("Model provided is schema version %d not equal to supported "
    // "version %d.", model->version(), TFLITE_SCHEMA_VERSION);
    return 1;
  }

  // Declare the op resolver and register only the necessary operations
  static tflite::MicroMutableOpResolver<9>
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
  // resolver.AddSoftmax();

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
  MicroPrintf("Input shape: %d x %d x %d", input->dims->data[1],
              input->dims->data[2], input->dims->data[3]);

  MicroPrintf("Loading picture data to input tensor. Length %lu", len);
  for (int i = 0; i < 144 * 144 * 3; i++) {
    input->data.int8[i] = ((int8_t *)ptr)[i];
  }

  MicroPrintf("Input type: %d", input->type);
  MicroPrintf("Input scale: %f, zero_point: %d", input->params.scale,
              input->params.zero_point);

  for (int i = 0; i < 20; i++) {
    MicroPrintf("input[%d] = %d", i, input->data.int8[i]);
  }

  // Run the model on this input and make sure it succeeds.
  if (kTfLiteOk != interpreter->Invoke()) {
    return 0.0f;
  }

  TfLiteTensor *output = interpreter->output(0);
  int8_t cat_score_q = output->data.int8[0];
  int8_t not_cat_score_q = output->data.int8[1];

  float scale = output->params.scale;
  int zero_point = output->params.zero_point;

  float cat_score = (cat_score_q - zero_point) * scale;
  float not_cat_score = (not_cat_score_q - zero_point) * scale;

  MicroPrintf("Cat: %f, Not Cat: %f", cat_score, not_cat_score);

  MicroPrintf("Raw output int8: cat = %d, not_cat = %d", cat_score_q,
              not_cat_score_q);

  float e_cat = expf(cat_score);
  float e_not_cat = expf(not_cat_score);
  float sum = e_cat + e_not_cat;

  float softmax_cat = e_cat / sum;
  float softmax_not_cat = e_not_cat / sum;

  MicroPrintf("Softmax → Cat: %.5f, Not Cat: %.5f", softmax_cat,
              softmax_not_cat);

  return cat_score;
}

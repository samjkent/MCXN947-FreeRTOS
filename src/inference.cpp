#include "cat.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <iostream>

constexpr int kTensorArenaSize = 100 * 1024; // 100KB (adjust as needed)
uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;

int init_model() {
  // Load model from memory
  model = tflite::GetModel(cat_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    // loge("Model provided is schema version %d not equal to supported "
    // "version %d.", model->version(), TFLITE_SCHEMA_VERSION);
    return 1;
  }

  // Declare the op resolver and register only the necessary operations
  static tflite::MicroMutableOpResolver<4>
      resolver; // Adjust NUM_OPS based on the model

  // Register required TensorFlow Lite operations used in your model
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddFullyConnected();
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
}

void run_inference(void *ptr) {
  /* Convert from uint8 picture data to int8 */
  for (int i = 0; i < 224 * 224; i++) {
    input->data.int8[i] = ((uint8_t *)ptr)[i] ^ 0x80;
  }

  // Run the model on this input and make sure it succeeds.
  if (kTfLiteOk != interpreter->Invoke()) {
    MicroPrintf("Invoke failed.");
  }

  TfLiteTensor *output = interpreter->output(0);

  // Process the inference results.
  int8_t person_score = output->data.uint8[0];
  int8_t no_person_score = output->data.uint8[1];

  float person_score_f =
      (person_score - output->params.zero_point) * output->params.scale;
  float no_person_score_f =
      (no_person_score - output->params.zero_point) * output->params.scale;
}

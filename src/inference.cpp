#include "cat.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <iostream>

// Define tensor arena size (adjust based on model requirements)
constexpr int kTensorArenaSize = 100 * 1024; // 100KB (adjust as needed)
uint8_t tensor_arena[kTensorArenaSize];

int init_model() {
  // Set up error reporter
  static tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter *error_reporter = &micro_error_reporter;

  // Load model from memory
  const tflite::Model *model = tflite::GetModel(tflite_model);
  if (model == nullptr) {
    // error_reporter->Report("Failed to load model!");
    return -1;
  }

  // Declare the op resolver and register only the necessary operations
  static tflite::MicroMutableOpResolver<1>
      resolver; // Adjust NUM_OPS based on the model

  // Register required TensorFlow Lite operations used in your model
  resolver.AddConv2D();
  // resolver.AddDepthwiseConv2D();
  // resolver.AddFullyConnected();
  // resolver.AddSoftmax();

  // FIX: Use correct MicroAllocator::Create() syntax
  tflite::MicroAllocator *allocator = tflite::MicroAllocator::Create(
      tensor_arena, kTensorArenaSize, tflite::MemoryPlannerType::kGreedy);

  if (!allocator) {
    // error_reporter->Report("Failed to create MicroAllocator!");
    return -1;
  }

  // FIX: Use the correct MicroInterpreter constructor with MicroAllocator
  static tflite::MicroInterpreter interpreter(model, resolver, allocator,
                                              nullptr, nullptr);

  // Allocate memory for tensors
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    // error_reporter->Report("Failed to allocate tensors!");
    return -1;
  }

  // Get input tensor
  TfLiteTensor *input_tensor = interpreter.input(0);
  if (!input_tensor) {
    // error_reporter->Report("Failed to get input tensor!");
    return -1;
  }

  // Ensure input shape matches expected [1, 244, 244, 3]
  if (input_tensor->dims->size != 4 || input_tensor->dims->data[1] != 244 ||
      input_tensor->dims->data[2] != 244 || input_tensor->dims->data[3] != 3) {
    // error_reporter->Report("Invalid input shape!");
    return -1;
  }

  // Fill input tensor with dummy data (replace with actual image data)
  float *input_data = input_tensor->data.f;
  for (int i = 0; i < 244 * 244 * 3; i++) {
    input_data[i] = 0.5f; // Example normalized pixel values
  }

  // Run inference
  if (interpreter.Invoke() != kTfLiteOk) {
    // error_reporter->Report("Failed to invoke the interpreter!");
    return -1;
  }

  // Get output tensor
  TfLiteTensor *output_tensor = interpreter.output(0);
  if (!output_tensor) {
    // error_reporter->Report("Failed to get output tensor!");
    return -1;
  }

  // Read output data (binary classification, so assume a single float output)
  float output_value = output_tensor->data.f[0];

  // Print the inference result
  // std::cout << "Inference Output: " << output_value << std::endl;

  return 0;
}

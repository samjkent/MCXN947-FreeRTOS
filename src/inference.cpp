#include "cat.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <iostream>

// Define tensor arena size (adjust based on model requirements)
constexpr int kTensorArenaSize = 100 * 1024; // 100KB (adjust as needed)
uint8_t tensor_arena[kTensorArenaSize];

int init_model() {
    // // Set up error reporter
    // // tflite::MicroErrorReporter micro_error_reporter;
    // // tflite::ErrorReporter *error_reporter = &micro_error_reporter;

    // // Load model from memory
    // const tflite::Model *model = tflite::GetModel(tflite_model);
    // if (model == nullptr) {
    //   // error_reporter->Report("Failed to load model!");
    //   return -1;
    // }

    // // Declare the op resolver and register only the necessary operations
    // static tflite::MicroMutableOpResolver<1>
    //   resolver; // Adjust NUM_OPS based on the model

    // // Register required TensorFlow Lite operations used in your model
    // resolver.AddConv2D();
    // // resolver.AddDepthwiseConv2D();
    // // resolver.AddFullyConnected();
    // // resolver.AddSoftmax();
}

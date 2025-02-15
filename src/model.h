#ifndef MODEL_H
#define MODEL_H

#include "tensorflow/lite/micro/micro_interpreter.h"
#include <stdint.h>

// Shared memory buffer for the input image (updated by another task)
extern volatile uint8_t image_buffer[244][244][3];

// Function to initialize the TensorFlow model
void init_model();

// Function to run inference (returns classification result)
int run_inference();

#endif // MODEL_H

#ifndef MODEL_H
#define MODEL_H

#include <stdint.h>

// Shared memory buffer for the input image (updated by another task)
extern volatile uint8_t image_buffer[244][244][3];

void vInferenceTask(void *pvParameters);

#endif // MODEL_H

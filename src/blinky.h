#ifndef BLINKY_H
#define BLINKY_H

#include "fsl_gpio.h"

// LED Pin Definitions
#define LED_PORT       GPIO1
#define LED_PIN        2

// Function Prototypes
void LED_Init(void);
void LED_Toggle(void);
void vBlinkyTask(void *pvParameters);

#endif /* BLINKY_H */
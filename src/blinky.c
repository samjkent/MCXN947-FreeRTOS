#include "blinky.h"
#include "fsl_gpio.h"
#include "fsl_port.h"
#include "fsl_clock.h"
#include "fsl_common.h"
#include "FreeRTOS.h"
#include "task.h"

void LED_Init(void) {
    CLOCK_EnableClock(LED_CLOCK);

    gpio_pin_config_t gpio_config = {kGPIO_DigitalOutput, 0};
    GPIO_PinInit(LED_PORT, LED_PIN , &gpio_config);
}

void LED_Toggle(void) {
    GPIO_PortToggle(LED_PORT, 1 << LED_PIN);
}

void vBlinkyTask(void *pvParameters) {
    (void) pvParameters;

    LED_Init();

    while (1) {
        LED_Toggle();
        vTaskDelay(pdMS_TO_TICKS(500));
    }
}

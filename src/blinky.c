#include "blinky.h"
#include "FreeRTOS.h" // IWYU pragma: keep
#include "fsl_clock.h"
#include "fsl_gpio.h"
#include "task.h"

void LED_Init(void) {
  CLOCK_EnableClock(kCLOCK_Gpio1);

  gpio_pin_config_t gpio_config = {kGPIO_DigitalOutput, 0};
  GPIO_PinInit(LED_PORT, LED_PIN, &gpio_config);
}

void LED_Toggle(void) { GPIO_PortToggle(LED_PORT, 1 << LED_PIN); }

void vBlinkyTask(void *pvParameters) {
  LED_Init();

  while (1) {
    LED_Toggle();
    vTaskDelay(pdMS_TO_TICKS(500));
  }
}

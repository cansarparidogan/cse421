#ifndef __STM32746G_DISCOVERY_TSENSOR_H
#define __STM32746G_DISCOVERY_TSENSOR_H

#include "mbed.h"
#include "stm32f7xx_hal.h"

#ifdef __cplusplus
extern "C" {
#endif

void BSP_TSENSOR_Init(void);
float BSP_TSENSOR_ReadTemp(void);

#ifdef __cplusplus
}
#endif

#endif

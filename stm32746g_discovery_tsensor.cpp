#include "stm32746g_discovery_tsensor.h"

static I2C_HandleTypeDef I2cHandle;

#define TSENSOR_I2C_SCL_PIN       GPIO_PIN_8
#define TSENSOR_I2C_SDA_PIN       GPIO_PIN_9
#define TSENSOR_I2C_GPIO_PORT     GPIOB
#define TSENSOR_I2C_AF            GPIO_AF4_I2C1
#define TSENSOR_I2C_INSTANCE      I2C1
#define TSENSOR_I2C_ADDRESS       ((uint8_t)0x90)

void BSP_TSENSOR_Init(void)
{
    GPIO_InitTypeDef GPIO_InitStruct = {0};

    __HAL_RCC_GPIOB_CLK_ENABLE();
    __HAL_RCC_I2C1_CLK_ENABLE();

    GPIO_InitStruct.Pin = TSENSOR_I2C_SCL_PIN | TSENSOR_I2C_SDA_PIN;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_OD;
    GPIO_InitStruct.Pull = GPIO_PULLUP;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    GPIO_InitStruct.Alternate = TSENSOR_I2C_AF;
    HAL_GPIO_Init(TSENSOR_I2C_GPIO_PORT, &GPIO_InitStruct);

    I2cHandle.Instance = TSENSOR_I2C_INSTANCE;
    I2cHandle.Init.Timing = 0x40912732; // 100kHz I2C @216MHz
    I2cHandle.Init.OwnAddress1 = 0;
    I2cHandle.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
    I2cHandle.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
    I2cHandle.Init.OwnAddress2 = 0;
    I2cHandle.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
    I2cHandle.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;
    HAL_I2C_Init(&I2cHandle);
}

float BSP_TSENSOR_ReadTemp(void)
{
    uint8_t reg = 0x00;
    uint8_t data[2];

    HAL_I2C_Master_Transmit(&I2cHandle, TSENSOR_I2C_ADDRESS, &reg, 1, HAL_MAX_DELAY);
    HAL_I2C_Master_Receive(&I2cHandle, TSENSOR_I2C_ADDRESS, data, 2, HAL_MAX_DELAY);

    int16_t raw = ((int16_t)data[0] << 8) | data[1];
    float temp = (raw >> 7) * 0.5f;

    return temp;
}

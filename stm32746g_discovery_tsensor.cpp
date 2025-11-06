#include "mbed.h"
#include "stm32f7xx_hal.h"
#include "stm32746g_discovery_sdram.h"
#include "stm32746g_discovery_lcd.h"
#include "fonts.h"

BufferedSerial pc(USBTX, USBRX, 115200);
FileHandle *mbed::mbed_override_console(int){ return &pc; }
DigitalOut led(PI_1);

#define ADDR_TS_CAL1 ((uint16_t*)0x1FF0F44C)
#define ADDR_TS_CAL2 ((uint16_t*)0x1FF0F44E)
#define ADDR_VREFIN_CAL ((uint16_t*)0x1FF0F44A)

static ADC_HandleTypeDef hadc1;

static void adc_init() {
    __HAL_RCC_ADC1_CLK_ENABLE();
    ADC->CCR |= ADC_CCR_TSVREFE;
    hadc1.Instance = ADC1;
    hadc1.Init.ClockPrescaler = ADC_CLOCK_SYNC_PCLK_DIV4;
    hadc1.Init.Resolution = ADC_RESOLUTION_12B;
    hadc1.Init.ScanConvMode = DISABLE;
    hadc1.Init.ContinuousConvMode = DISABLE;
    hadc1.Init.DiscontinuousConvMode = DISABLE;
    hadc1.Init.NbrOfDiscConversion = 0;
    hadc1.Init.ExternalTrigConvEdge = ADC_EXTERNALTRIGCONVEDGE_NONE;
    hadc1.Init.ExternalTrigConv = ADC_SOFTWARE_START;
    hadc1.Init.DataAlign = ADC_DATAALIGN_RIGHT;
    hadc1.Init.NbrOfConversion = 1;
    hadc1.Init.DMAContinuousRequests = DISABLE;
    hadc1.Init.EOCSelection = ADC_EOC_SINGLE_CONV;
    HAL_ADC_Init(&hadc1);
}

static uint16_t adc_read_channel(uint32_t ch, uint32_t samp = ADC_SAMPLETIME_480CYCLES) {
    ADC_ChannelConfTypeDef s{};
    s.Channel = ch;
    s.Rank = ADC_REGULAR_RANK_1;
    s.SamplingTime = samp;
    HAL_ADC_ConfigChannel(&hadc1, &s);
    HAL_ADC_Start(&hadc1);
    HAL_ADC_PollForConversion(&hadc1, HAL_MAX_DELAY);
    uint16_t v = (uint16_t)HAL_ADC_GetValue(&hadc1);
    HAL_ADC_Stop(&hadc1);
    return v;
}

static float read_temp_c() {
    uint16_t vref_adc = adc_read_channel(ADC_CHANNEL_VREFINT);
    float vref_cal = (float)(*ADDR_VREFIN_CAL);
    float vref_corr = vref_cal / (float)vref_adc;
    uint16_t ts_adc = adc_read_channel(ADC_CHANNEL_TEMPSENSOR);
    float ts_corr = (float)ts_adc * vref_corr;
    float t30 = (float)(*ADDR_TS_CAL1);
    float t110 = (float)(*ADDR_TS_CAL2);
    float tempC = (ts_corr - t30) * (110.0f - 30.0f) / (t110 - t30) + 30.0f;
    return tempC;
}

int main() {
    BSP_SDRAM_Init();
    BSP_LCD_Init();
    BSP_LCD_LayerDefaultInit(0, LCD_FB_START_ADDRESS);
    BSP_LCD_SelectLayer(0);
    BSP_LCD_DisplayOn();
    BSP_LCD_Clear(LCD_COLOR_BLACK);
    BSP_LCD_SetBackColor(LCD_COLOR_BLACK);
    BSP_LCD_SetTextColor(LCD_COLOR_YELLOW);
    BSP_LCD_SetFont(&Font12);
    BSP_LCD_DisplayStringAt(0, 8, (uint8_t*)"CSE421 - Internal Temp x10", CENTER_MODE);
    adc_init();
    ThisThread::sleep_for(50ms);
    float temps[10], sum = 0.f;
    BSP_LCD_SetTextColor(LCD_COLOR_CYAN);
    BSP_LCD_DisplayStringAt(0, 26, (uint8_t*)"Sampling 10 readings...", CENTER_MODE);
    for (int i = 0; i < 10; ++i) {
        led = !led;
        float t = read_temp_c();
        temps[i] = t; sum += t;
        int tempInt = (int)roundf(t);
        printf("T%d,%d\r\n", i+1, tempInt); fflush(stdout);
        char msg[48];
        sprintf(msg, "Reading %d/10: %d C", i+1, tempInt);
        BSP_LCD_SetTextColor(LCD_COLOR_YELLOW);
        BSP_LCD_DisplayStringAt(0, 42, (uint8_t*)msg, CENTER_MODE);
        ThisThread::sleep_for(300ms);
    }
    BSP_LCD_Clear(LCD_COLOR_BLACK);
    BSP_LCD_SetTextColor(LCD_COLOR_YELLOW);
    BSP_LCD_DisplayStringAt(0, 8, (uint8_t*)"10 Temperature Readings", CENTER_MODE);
    BSP_LCD_SetTextColor(LCD_COLOR_GREEN);
    int y = 24; const int dy = 14;
    for (int i = 0; i < 10; ++i) {
        int tempInt = (int)roundf(temps[i]);
        char row[40];
        sprintf(row, "T%-2d: %d C", i+1, tempInt);
        BSP_LCD_DisplayStringAt(8, y, (uint8_t*)row, LEFT_MODE);
        y += dy;
    }
    float avg = sum / 10.f;
    int avgInt = (int)roundf(avg);
    BSP_LCD_SetTextColor(LCD_COLOR_CYAN);
    char avgLine[40];
    sprintf(avgLine, "AVG: %d C", avgInt);
    BSP_LCD_DisplayStringAt(0, y + 6, (uint8_t*)avgLine, CENTER_MODE);
    while (true) { led = !led; ThisThread::sleep_for(600ms); }
}

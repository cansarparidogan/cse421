#include "mbed.h"
#include "stm32746g_discovery_sdram.h"
#include "stm32746g_discovery_lcd.h"
#include "fonts.h"

#define VDDA 3.3f
#define R_FIXED 10000.0f
#define OVERSAMPLE 64
#define USE_LUX 1

BufferedSerial pc(USBTX, USBRX, 115200);
FileHandle *mbed::mbed_override_console(int){ return &pc; }
DigitalOut led(PI_1);
AnalogIn ain(PA_0);

static float read_v() {
    uint32_t acc=0;
    for (int i=0;i<OVERSAMPLE;i++) acc += ain.read_u16();
    float avg = acc / (float)OVERSAMPLE;
    return (avg/65535.0f)*VDDA;
}

#if USE_LUX
static float ldr_resistance(float vout){
    if (vout<=0.001f) vout=0.001f;
    if (vout>=VDDA-0.001f) vout=VDDA-0.001f;
    return R_FIXED*(vout/(VDDA - vout));
}
static float approx_lux(float r){
    const float A=1000.0f, G=1.4f;
    return A*powf(r, -G);
}
#endif

int main(){
    BSP_SDRAM_Init();
    BSP_LCD_Init();
    BSP_LCD_LayerDefaultInit(0, LCD_FB_START_ADDRESS);
    BSP_LCD_SelectLayer(0);
    BSP_LCD_DisplayOn();
    BSP_LCD_Clear(LCD_COLOR_BLACK);
    BSP_LCD_SetBackColor(LCD_COLOR_BLACK);
    BSP_LCD_SetTextColor(LCD_COLOR_YELLOW);
    BSP_LCD_SetFont(&Font12);
    BSP_LCD_DisplayStringAt(0, 8, (uint8_t*)"Part 2 - Light/Pot ADC (x10)", CENTER_MODE);

    float v[10], m[10], sumV=0, sumM=0;

    BSP_LCD_SetTextColor(LCD_COLOR_CYAN);
    BSP_LCD_DisplayStringAt(0, 26, (uint8_t*)"Sampling 10 readings...", CENTER_MODE);

    for (int i=0;i<10;i++){
        led=!led;
        float vo = read_v();
        v[i]=vo; sumV+=vo;

    #if USE_LUX
        float r = ldr_resistance(vo);
        float metric = approx_lux(r);
        m[i]=metric; sumM+=metric;
        printf("N%d,%.1flux\r\n", i+1, metric); fflush(stdout);
        char line[48]; sprintf(line,"N%02d: %.1flux", i+1, metric);
    #else
        float metric = (vo/VDDA)*100.0f;
        m[i]=metric; sumM+=metric;
        printf("N%d,%.1f%%\r\n", i+1, metric); fflush(stdout);
        char line[48]; sprintf(line,"N%02d: %.1f%%", i+1, metric);
    #endif

        BSP_LCD_SetTextColor(LCD_COLOR_YELLOW);
        BSP_LCD_DisplayStringAt(0, 42, (uint8_t*)line, CENTER_MODE);
        ThisThread::sleep_for(300ms);
    }

    BSP_LCD_Clear(LCD_COLOR_BLACK);
    BSP_LCD_SetTextColor(LCD_COLOR_YELLOW);
    BSP_LCD_DisplayStringAt(0, 8, (uint8_t*)"10 Readings", CENTER_MODE);

    BSP_LCD_SetTextColor(LCD_COLOR_GREEN);
    int y=24, dy=14;
    for (int i=0;i<10;i++){
        char row[48];
        sprintf(row,"N%02d: %.1f", i+1, m[i]);
        BSP_LCD_DisplayStringAt(8, y, (uint8_t*)row, LEFT_MODE);
        y+=dy;
    }

    float avgM = sumM/10.0f;
    BSP_LCD_SetTextColor(LCD_COLOR_CYAN);
    char avgLine[48];
    sprintf(avgLine, "AVG: %.1f", avgM);
    BSP_LCD_DisplayStringAt(0, y+6, (uint8_t*)avgLine, CENTER_MODE);

    while (true){ led=!led; ThisThread::sleep_for(600ms); }
}

#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void mandelbrot(__global float *results,
                         unsigned int width, unsigned int height,
                         float fromX, float fromY,
                         float sizeX, float sizeY,
                         unsigned int iters,
                         unsigned int smoothing) {

    const float threshold = 256.0f;
    const float threshold2 = threshold * threshold;

    const unsigned int idx = get_global_id(0);
    const unsigned int idy = get_global_id(1);

    if (idx >= width || idy >= height) { return; }
    
    float x0 = fromX + (idx + 0.5f) * sizeX / width;
    float y0 = fromY + (idy + 0.5f) * sizeY / height;

    float x = x0;
    float y = y0;

    unsigned int iter = 0;

    for (; iter < iters; ++iter) {
        float xPrev = x;
        x = x * x - y * y + x0;
        y = 2.0f * xPrev * y + y0;
        if ((x * x + y * y) > threshold2) {
            break;
        }
    }

    float result = iter;

    if ((smoothing == 1) && (iter != iters)) {
        result = result - native_log(
            native_log(native_sqrt(x * x + y * y)) / 
            native_log(threshold)
        ) / native_log(2.0f);
    }

    result = 1.0f * result / iters;
    results[idy * width + idx] = result;

    // TODO если хочется избавиться от зернистости и дрожания при интерактивном погружении, добавьте anti-aliasing:
    // грубо говоря, при anti-aliasing уровня N вам нужно рассчитать не одно значение в центре пикселя, а N*N значений
    // в узлах регулярной решетки внутри пикселя, а затем посчитав среднее значение результатов - взять его за результат для всего пикселя
    // это увеличит число операций в N*N раз, поэтому при рассчетах гигаплопс антиальясинг должен быть выключен
    return;
}

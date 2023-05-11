#include <CL/cl.h>
#include <libclew/ocl_init.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>


template<typename T>
std::string to_string(T value) {
    std::ostringstream ss;
    ss << value;
    return ss.str();
}

std::string cl_device_type_to_string(cl_device_type type) {
    switch (type)
    {
    case CL_DEVICE_TYPE_CPU:
        return to_string("CPU");
    case CL_DEVICE_TYPE_GPU:
        return to_string("GPU");
    default:
        return to_string("Another");
    }
}

void reportError(cl_int err, const std::string &filename, int line) {
    if (CL_SUCCESS == err)
        return;

    // Таблица с кодами ошибок:
    // libs/clew/CL/cl.h:103
    // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
    std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
    throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)


int main() {
    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку libs/clew)
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    // Откройте
    // https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/
    // Нажмите слева: "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformIDs"
    // Прочитайте документацию clGetPlatformIDs и убедитесь, что этот способ узнать, сколько есть платформ, соответствует документации:
    cl_uint platformsCount = 0;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
    std::cout << "Number of OpenCL platforms: " << platformsCount << std::endl;

    // Тот же метод используется для того, чтобы получить идентификаторы всех платформ - сверьтесь с документацией, что это сделано верно:
    std::vector<cl_platform_id> platforms(platformsCount);
    OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));

    for (int platformIndex = 0; platformIndex < platformsCount; ++platformIndex) {
        std::cout << "Platform #" << (platformIndex + 1) << "/" << platformsCount << std::endl;
        cl_platform_id platform = platforms[platformIndex];

        // Откройте документацию по "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformInfo"
        // Не забывайте проверять коды ошибок с помощью макроса OCL_SAFE_CALL
        size_t platformNameSize = 0;
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platformNameSize));
        // OCL_SAFE_CALL(clGetPlatformInfo(platform, 4, 0, nullptr, &platformNameSize));
        // TODO 1.1
        // Попробуйте вместо CL_PLATFORM_NAME передать какое-нибудь случайное число - например 239
        // Т.к. это некорректный идентификатор параметра платформы - то метод вернет код ошибки
        // Макрос OCL_SAFE_CALL заметит это, и кинет ошибку с кодом
        // Откройте таблицу с кодами ошибок:
        // libs/clew/CL/cl.h:103
        // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
        // Найдите там нужный код ошибки и ее название
        // Затем откройте документацию по clGetPlatformInfo и в секции Errors найдите ошибку, с которой столкнулись
        // в документации подробно объясняется, какой ситуации соответствует данная ошибка, и это позволит, проверив код, понять, чем же вызвана данная ошибка (некорректным аргументом param_name)
        // Обратите внимание, что в этом же libs/clew/CL/cl.h файле указаны всевоможные defines, такие как CL_DEVICE_TYPE_GPU и т.п.

        // TODO 1.2
        // Аналогично тому, как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:
        std::vector<unsigned char> platformName(platformNameSize, 0);
        // clGetPlatformInfo(...);
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), &platformNameSize));
        std::cout << "    Platform name: " << platformName.data() << std::endl;

        // TODO 1.3
        // Запросите и напечатайте так же в консоль вендора данной платформы
        size_t platformVendorSize = 0;
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &platformVendorSize));
        std::vector<unsigned char> platformVendor(platformVendorSize, 0);
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, platformVendorSize, platformVendor.data(), 0));
        std::cout << "    Platform vendor: " << platformVendor.data() << std::endl;

        // TODO 2.1
        // Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
        cl_uint devicesCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
        std::cout << "      Number of platform devices: " << devicesCount << std::endl;

        std::vector<cl_device_id> devices(devicesCount);
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), &devicesCount));


        for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
            std::cout << "      Device #" << (deviceIndex + 1) << "/" << devicesCount << std::endl;
            // TODO 2.2
            // Запросите и напечатайте в консоль:
            cl_device_id device = devices[deviceIndex];

            // - Название устройства
            size_t deviceNameSize = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &deviceNameSize));

            std::vector<unsigned char> deviceName(deviceNameSize, 0);
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, deviceNameSize, deviceName.data(), &deviceNameSize));
            std::cout << "          Device name: " << deviceName.data() << std::endl;

            // - Тип устройства (видеокарта/процессор/что-то странное)
            cl_device_type deviceType;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &deviceType, nullptr));
            std::cout << "          Device type: " << cl_device_type_to_string(deviceType) << std::endl;

            // - Размер памяти устройства в мегабайтах
            cl_ulong deviceGlobalMemCacheSize = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(cl_ulong), &deviceGlobalMemCacheSize, nullptr));
            std::cout << "          Device global mem cache size: " << (deviceGlobalMemCacheSize >> 10) << " KB" << std::endl;

            cl_ulong deviceGlobalMemCachelineSize = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(cl_ulong), &deviceGlobalMemCachelineSize, nullptr));
            std::cout << "          Device global mem cacheline size: " << (deviceGlobalMemCachelineSize >> 0) << " B" << std::endl;

            cl_ulong deviceGlobalMemSize = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &deviceGlobalMemSize, nullptr));
            std::cout << "          Device global mem size: " << (deviceGlobalMemSize >> 20) << " MB" << std::endl;

            cl_ulong deviceLocalMemSize = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &deviceLocalMemSize, nullptr));
            std::cout << "          Device local mem size: " << (deviceLocalMemSize >> 10) << " KB" << std::endl;

            // - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными

            size_t deviceMaxWorkGroupSize = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &deviceMaxWorkGroupSize, nullptr));
            std::cout << "          Device max work group size: " << (deviceMaxWorkGroupSize) << "" << std::endl;

            cl_ulong deviceMaxWorkItemDim = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_ulong), &deviceMaxWorkItemDim, nullptr));
            std::cout << "          Device max work item dimensions: " << (deviceMaxWorkItemDim) << "" << std::endl;

            std::vector<size_t> deviceMaxWorkItemSizes(deviceMaxWorkItemDim, 0);
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(cl_ulong) * deviceMaxWorkItemDim, deviceMaxWorkItemSizes.data(), nullptr));
            std::cout << "          Device max work group sizes: ";
            for (size_t i = 0; i < deviceMaxWorkItemDim; i++) {
                std::cout << deviceMaxWorkItemSizes[i] << " ";
            } 
            std::cout << std::endl;
        }
    }
    return 0;
}

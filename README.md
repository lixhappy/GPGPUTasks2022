В этом репозитории предложены задания для В этом репозитории предложены задания для курса по вычислениям на видеокартах 2022.

[Остальные задания](https://github.com/GPGPUCourse/GPGPUTasks2022/).

# Задание 1. A+B.

[![Build Status](https://github.com/lixhappy/GPGPUTasks2022/actions/workflows/cmake.yml/badge.svg?branch=task01&event=push)](https://github.com/lixhappy/GPGPUTasks2022/actions/workflows/cmake.yml)

Задание
=======

0. Сделать fork проекта
1. Прочитать все комментарии подряд и выполнить все **TODO** в файле ``src/main.cpp`` и ``src/cl/aplusb.cl``
2. Отправить **Pull-request** с названием```Task01 <Имя> <Фамилия> <Аффиляция>``` (добавив в описании вывод работы программы в **pre**-тэгах - см. [пример](https://raw.githubusercontent.com/GPGPUCourse/GPGPUTasks2022/task01/.github/pull_request_example.md))

**Дедлайн**: начало лекции 27 сентября.

Коментарии
==========

Т.к. в ``TODO 6`` исходники кернела считываются по относительному пути ``src/cl/aplusb.cl``, то нужно правильно настроить working directory. Например в случае CLion нужно открыть ``Edit configurations`` -> и указать ``Working directory: .../НАЗВАНИЕПАПКИПРОЕКТА`` (см. [подробнее](https://github.com/lixhappy/GPGPUTasks2022/tree/task01/.figures))

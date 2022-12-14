QT += core
QT -= gui

CONFIG += c++11

TARGET = FisheyeStitch
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += main.cpp \
    undistort.cpp \
    test/fisheye_stitcher.cpp \
#    test/stitch.cpp
    fisheyestitcher.cpp \
    threadpool/thread_pool.cpp \
    test/ImageStitch.cpp

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

HEADERS += \
    undistort.h \
    test/fisheye_stitcher.hpp \
#    test/input_parser.hpp
    fisheyestitcher.h \
    threadpool/function_wrapper.h \
    threadpool/join_threads.h \
    threadpool/thread_pool.h \
    threadpool/threadsafe_queue.h \
    test/ImageStitch.h \
    test/multibandblendertest.hpp

INCLUDEPATH += /usr/local/include/opencv4/ \
            += /usr/local/include/opencv4/opencv2/ \
            += /usr/include/eigen3/

LIBS += /usr/local/lib/libopencv*.so

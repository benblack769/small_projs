#-------------------------------------------------
#
# Project created by QtCreator 2016-04-09T12:02:11
#
#-------------------------------------------------

QT       = core

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = path
TEMPLATE = app

QMAKE_CXXFLAGS += -std=c++11
QMAKE_CXXFLAGS_RELEASE -= -O2
QMAKE_CXXFLAGS_RELEASE += -O3 -march=native -mtune=native

SOURCES += main.cpp\
        mainwindow.cpp

HEADERS  += mainwindow.h \
    board.h \
    point.h \
    farray2d.h \
    walk.h

FORMS    += mainwindow.ui

DISTFILES += \
    .gitignore

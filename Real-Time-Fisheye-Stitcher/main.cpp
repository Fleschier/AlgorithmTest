#include "mainwindow.h"
#include <QApplication>

#include "stitcher.h"
#include "HistgramMatch/HistogramMatching.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.show();

    std::thread t = std::thread(
                []()->void{
                    FishEyeStitcher mStitcher;
                    if(!mStitcher.Init()){
                        printf("error while init stitcher!\n return ...");
                        exit(-1);
                    }
                    printf("test generate\n");
                    //mStitcher.TestGenerate(2);
                    mStitcher.TestGenerateVideo();
                    //maintest("./HistoTset/reference.jpg", "./HistoTset/input.jpg", "./reverseTest.jpg");

                });
    if(t.joinable())
        t.detach();

    return a.exec();
}

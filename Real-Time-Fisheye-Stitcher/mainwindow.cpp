#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "stitcher.h"

MainWindow::MainWindow(QWidget *parent) :
  QMainWindow(parent),
  ui(new Ui::MainWindow)
{
  ui->setupUi(this);
  FishEyeStitcher mStitcher;
  if(!mStitcher.Init()){
      printf("error while init stitcher!\n return ...");
      exit(-1);
    }
  printf("test generate\n");
//  mStitcher.TestGenerate(2);
  mStitcher.TestGenerateVideo();
}

MainWindow::~MainWindow()
{
  delete ui;
}

void MainWindow::on_pushButton_clicked()
{

}

void MainWindow::on_pushButton_2_clicked()
{

}

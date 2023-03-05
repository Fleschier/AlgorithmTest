#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "stitcher.h"
#include "HistgramMatch/HistogramMatching.h"
#include<chrono>

using namespace std;
using namespace chrono;
using namespace cv;

MainWindow::MainWindow(QWidget *parent) :
  QMainWindow(parent),
  ui(new Ui::MainWindow)
{
  ui->setupUi(this);

  connect(&theTimer, &QTimer::timeout, this, &MainWindow::updateImage);
  if(videoCap.open("/home/fleschier/programes/Videos/gear360/360_0108_pano_comp.mp4"))
      {
          srcImage = Mat::zeros(videoCap.get(cv::CAP_PROP_FRAME_HEIGHT), videoCap.get(cv::CAP_PROP_FRAME_WIDTH), CV_8UC3);
          theTimer.start(33);
      }
  ui->label->setScaledContents(true);

//  FishEyeStitcher mStitcher;
//  if(!mStitcher.Init()){
//      printf("error while init stitcher!\n return ...");
//      exit(-1);
//    }
//  printf("test generate\n");
  //mStitcher.TestGenerate(2);
  //mStitcher.TestGenerateVideo();
  //maintest("./HistoTset/reference.jpg", "./HistoTset/input.jpg", "./reverseTest.jpg");

//  cv::namedWindow("test", cv::WINDOW_FULLSCREEN);
//  for(int i = 0; i < 30; i++)
//  {
//      videoCap.read(srcImage);
//      auto begin = system_clock::now();
//      imshow("test", srcImage);
//      auto end = system_clock::now();
//      auto duration = duration_cast<microseconds>(end - begin);
//      cout << "one frame light compensation spends: "
//           << double(duration.count()) * microseconds::period::num / microseconds::period::den
//           << "seconds" << endl;
//  }
}

MainWindow::~MainWindow()
{
  delete ui;
}

void MainWindow::paintEvent(QPaintEvent *e)
{
    auto begin = system_clock::now();
//    //显示方法一
//    QPainter painter(this);
//    QImage image1 = QImage((uchar*)(srcImage.data), srcImage.cols, srcImage.rows, QImage::Format_RGB888);
//    painter.drawImage(QPoint(20,20), image1);
    //显示方法二
    QImage image2 = QImage((uchar*)(srcImage.data), srcImage.cols, srcImage.rows, QImage::Format_RGB888);
    ui->label->setPixmap(QPixmap::fromImage(image2));
    //ui->label->resize(image2.size());
    ui->label->show();
    auto end = system_clock::now();
    auto duration = duration_cast<microseconds>(end - begin);
    cout << "one frame light compensation spends: "
         << double(duration.count()) * microseconds::period::num / microseconds::period::den
         << "seconds" << endl;
}

void MainWindow::updateImage()
{
    videoCap>>srcImage;
    if(srcImage.data)
    {
        cvtColor(srcImage, srcImage, COLOR_BGR2RGB);//Qt中支持的是RGB图像, OpenCV中支持的是BGR
        this->update();  //发送刷新消息
    }
}

void MainWindow::on_pushButton_clicked()
{

}

void MainWindow::on_pushButton_2_clicked()
{

}

void MainWindow::on_pushButton_3_clicked()
{

}

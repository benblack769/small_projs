#pragma once

#include <QMainWindow>
#include <vector>
#include "ui_mainwindow.h"
#include "point.h"
#include "farray2d.h"

extern double rand_lin_val;
extern double dest_lin_val;
extern double avoid_prev_lin_val;
extern int walk_dis;
extern bool is_lin_rand_walk;

using namespace std;

namespace Ui {
class MainWindow;
}
class QGraphicsView;
class QGraphicsScene;
class QGraphicsPixmapItem;

class MainWindow : public QMainWindow
{
    Q_OBJECT
    
public:
    explicit MainWindow(string filename, QWidget *parent = 0);
    ~MainWindow() = default;
    
    void draw_maze(FArray2d<char> maze);
    void draw_connected_pts(vector<QPointF> points,QPen pen);
    void draw_image(const QImage & img);
private:
    Ui::MainWindow * ui;
    QGraphicsView * view;
    QGraphicsScene * screen;
    QGraphicsPixmapItem * mazepix;
};

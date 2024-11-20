#include "mainwindow.h"

#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsLineItem>
#include "board.h"

MainWindow::MainWindow(string filename,QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow())
{
    
    string out_filename = "out"+filename;
    FArray2d<char> maze = make_maze(filename);
    Point mdim = maze.dim();
    
    ui->setupUi(this);
    
    
    screen = new QGraphicsScene;
    
    screen->setSceneRect(QRectF(0,0,mdim.X,mdim.Y));
    
    view = new QGraphicsView(screen);
    
    setCentralWidget(view);
    
    
    //draws objects
    draw_maze(maze);
    
    Point lastp = mdim-Point(1,1);
    
    //countinuous drawing
    vector<Point> dis_points = discrite_path_to_best(maze,Point(0,0),Point(0,0),lastp,[&](Point p){return double(p == lastp);});
    //vector<QPointF> pts = continuous_path(maze,dis_points);
    draw_connected_pts(conv_vec(dis_points),QPen(QBrush(Qt::red),2.0));
    //cout << "Smallest possible distance: " << connected_distance(pts) << endl;
    cout << connected_distance(conv_vec(dis_points)) << "\t";
    cout.flush();
    
    double dis = 0;
    vector<Point> rand_walk_path = 
            is_lin_rand_walk ? 
                rand_liniar_walk(maze,Point(0,0),lastp) : 
                rand_walk(maze,Point(0,0),lastp);
    
    draw_image(gen_QImage_density(make_density(mdim,rand_walk_path),Qt::blue));
    cout << connected_distance(conv_vec(rand_walk_path)) << endl;
        
    screen->clearSelection();                                     // Selections would also render to the file
    screen->setSceneRect(screen->itemsBoundingRect());                          // Re-shrink the scene to it's bounding contents
    QImage image(screen->sceneRect().size().toSize(), QImage::Format_ARGB32);  // Create the image with the exact size of the shrunk scene
    image.fill(Qt::transparent);                                              // Start all pixels transparent
    
    QPainter painter(&image);
    screen->render(&painter);
    image.save(out_filename.c_str());
}
void MainWindow::draw_maze(FArray2d<char> maze){
    mazepix = new QGraphicsPixmapItem(QPixmap::fromImage(gen_QImage_Maze(maze)));
    screen->addItem(mazepix);
}

void MainWindow::draw_connected_pts(vector<QPointF> points,QPen pen){
    for(size_t i = 1; i < points.size(); i++){
        auto line = screen->addLine(QLineF(points[i-1],points[i]),pen);
    }
}
void MainWindow::draw_image(const QImage & img){
    QGraphicsPixmapItem *img_pix = new QGraphicsPixmapItem(QPixmap::fromImage(img));
    screen->addItem(img_pix);
}

/*
int breadth_first_search_dis(FArray2d<char> maze){
    Point endp = maze.dim()-Point(1,1);
    int enddis = -1;
    breadth_first_search(maze,Point(0,0),endp,[&](Point p,int dist){
        if(p == endp){
            enddis = dist;
        }
    });
    assert(enddis >= 0 && "bfs failed to find end");
    return enddis;
}*/


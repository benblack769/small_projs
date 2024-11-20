#pragma once
#include <QPoint>
#include <array>
#include <vector>
#include <QPixmap>
#include <QImage>
#include <cassert>
#include <QRgb>
#include <unordered_set>
#include <unordered_map>
#include <cstdlib>
#include <cmath>
#include "point.h"
#include "farray2d.h"
#include "walk.h"
using namespace std;

bool is_white(QRgb col){
    return qRed(col) + qGreen(col) + qBlue(col) > (255*3)/2;
}
QImage gen_QImage_Maze(FArray2d<char> blocked_points){
    Point dim = blocked_points.dim();
    QImage img(dim.X,dim.Y,QImage::Format_ARGB32);
    for(int y = 0; y < dim.Y; y++){
        for(int x = 0; x < dim.X; x++){
            QColor col = (blocked_points[y][x]==0)?
                Qt::white :
                Qt::black;
            
            img.setPixelColor(x,y,col);
            
        }
    }
    return img;
}

FArray2d<int> make_density(Point dim,const vector<Point> & points){
    FArray2d<int> res(dim.X,dim.Y);
    res.assign(0);
    for(Point p : points){
        res[p]++;
    }
    return res;
}

QImage gen_QImage_density(FArray2d<int> pointvals,QColor overall_col){
    Point dim = pointvals.dim();
    QImage img(dim.X,dim.Y,QImage::Format_ARGB32);
    
    int maxp = max(1,*max_element(pointvals.begin(),pointvals.end()));
    
    for(int y = 0; y < dim.Y; y++){
        for(int x = 0; x < dim.X; x++){
            double fullval =254.0; 
            double alpha = (fullval*pointvals[y][x]) / maxp;
            
            QColor col = overall_col;
            col.setAlpha(int(alpha));
            
            img.setPixelColor(x,y,col);
            
        }
    }
    return img;
}

void gen_image(QString outname,FArray2d<char> blocked_points){
    gen_QImage_Maze(blocked_points).save(outname);
}

FArray2d<char> make_maze(string image_file){
    QString fname = QString::fromStdString(image_file);
    QPixmap piximg(fname);
    QImage img = piximg.toImage();
    QSize size = img.size();
    FArray2d<char> res(size.width(),size.height());
    for(int y = 0; y < size.height(); y++){
        for(int x = 0; x < size.width(); x++){
            QRgb col = img.pixel(x,y);
            res[y][x] = !is_white(col);
        }
    }
    return res;
}
template<typename fn_ty>
void iter_around(Point min_edge,Point max_edge,Point cen,fn_ty fn){
    for(int y = max(cen.Y-1,min_edge.Y); y <= min(cen.Y+1,max_edge.Y); y++){
        for(int x = max(cen.X-1,min_edge.X); x <= min(cen.X+1,max_edge.X); x++){
            fn(Point(x,y));
        }
    }
}
const Point null_pt = Point(-1,-1);
template<typename fn_ty>
void breadth_first_search(const FArray2d<char> & blocked_points,Point start,Point min_edge,Point max_edge,fn_ty fn){
    vector<Point> prevps;
    vector<Point> curps;
    unordered_set<Point> accessed_pts;
    
    auto add_point = [&](Point prevp,Point newp,int dis){
        accessed_pts.insert(newp);
        curps.push_back(newp);
        fn(prevp,newp,dis);
    };
    auto point_valid = [&](Point p){
        return !accessed_pts.count(p) && !blocked_points[p];
    };
    
    int distance = 0;
    add_point(null_pt,start,distance);
    do{
        prevps.swap(curps);
        curps.resize(0);
        
        for(Point pp : prevps){            
            iter_around(min_edge,max_edge,pp,[&](Point newp){
                if(point_valid(newp)){
                    add_point(pp,newp,distance);
                }
            });
        }
        
        distance++;
    }while(curps.size());
}
template<typename max_fn_ty>
vector<Point> discrite_path_to_best(const FArray2d<char> & blocked_points,Point start,Point min_edge,Point max_edge,max_fn_ty max_fn){
    unordered_map<Point,Point> prev_ps;
    double maxval = -10e50;
    Point maxp = null_pt;
    
    breadth_first_search(blocked_points,start,min_edge,max_edge,[&](Point prevp,Point newp,int ){
        double val = max_fn(newp);
        prev_ps[newp] = prevp;
        if(maxval < val){
            maxval = val;
            maxp = newp;
        }
    });
    vector<Point> back_track;
    Point curp = maxp;
    while(curp != null_pt){
        back_track.push_back(curp);
        curp = prev_ps[curp];
    }
    vector<Point> forward_path(back_track.rbegin(),back_track.rend());
    return forward_path;
}

vector<QPointF> continuous_path(const FArray2d<char> & blocked_points,const vector<Point> & path){
    vector<QPointF> res;
    size_t startidx = 0;
    res.push_back(middle_of(path[startidx]));
    for(size_t i = 1; i < path.size(); i++){
        QPointF start = middle_of(path[startidx]);
        QPointF curp = middle_of(path[i]);
        if(i > startidx + 1 && !direct_path_clear(blocked_points,curp,start)){
            res.push_back(start);
            startidx = i - 1;
        }
    } 
    if(my_point(res.back()) != path.back()){
        res.push_back(middle_of(path.back()));
    }
    return res;
}
double rand_lin_val = 1.0;
double dest_lin_val = 0.0001;
double avoid_prev_lin_val = 0.3;
int walk_dis = 4;
bool is_lin_rand_walk = false;
int stride_sqrd(){return sqr(walk_dis)-1;}

struct loc_val{
    Point origcen;
    Point dest;
    QPointF cen;
    QPointF lin_vec;
    loc_val(Point incen,Point indest){
        origcen = incen;
        dest = indest;
        cen = q_pt(incen);
        lin_vec = QPointF(0,0);
    }
    void add_lin(Point p,double val){
        if(p != origcen){
            QPointF qp = q_pt(p);
            double dis = distance(qp,cen);
            double dis_adj_val = val / max(1.0,dis);
            lin_vec += (qp - cen) * dis_adj_val;
        }
    }
    double point_val(Point p){
        if(p == dest){
            return 10e20;
        }
        else if(sqr(p.X-origcen.X) + sqr(p.Y-origcen.Y) <= stride_sqrd()){
            QPointF p_offset = q_pt(p) - cen;
            return QPointF::dotProduct(lin_vec,p_offset);
        }
        else{
            return -10000.0;
        }
    }
    double operator()(Point p){return point_val(p);}
};

QPointF rand_dir_point(QPointF cen){
    double theta = (rand()/double(RAND_MAX))*M_PI*2;
    double r = 1000.0;
    QPointF offset(r*cos(theta),r*sin(theta));
    return cen + offset;
}

//double avoid_val = 0;
//double lin_val = 0;
//double lin_val = 0;
loc_val gen_loc_val(Point cen,Point dest,Point prevp,Point back2p){
    loc_val lval(cen,dest);
    lval.add_lin(dest,dest_lin_val);
    lval.add_lin(prevp,avoid_prev_lin_val);
    lval.add_lin(back2p,avoid_prev_lin_val);
    lval.add_lin(my_point(rand_dir_point(q_pt(cen))),rand_lin_val);
    return lval;
}
template<typename bin_op_fn>
Point element_wise(Point one,Point other,bin_op_fn bop){
    return Point(bop(one.X,other.X),bop(one.Y,other.Y));
}

vector<Point> path_to_max(const FArray2d<char> & blocked_points,Point cen,loc_val lval){
    vector<Point> res;
    Point dim = blocked_points.dim();
    Point lastp = dim - Point(1,1);
    Point min_edge = element_wise(Point(0,0),cen-Point(1,1)*walk_dis,[](int a,int b){return max(a,b);});
    Point max_edge = element_wise(lastp,     cen+Point(1,1)*walk_dis,[](int a,int b){return min(a,b);});
       
    return discrite_path_to_best(blocked_points,cen,min_edge,max_edge,lval);
}

vector<Point> rand_liniar_walk(const FArray2d<char> & blocked_points,Point begin, Point end){
    vector<Point> res;
    Point back2p = begin;
    Point prevp = begin;
    Point curp = begin;
    int count = 1;
    while(curp != end){
        if(count % 10000 == 0){
            //cout << count << endl;
            //cout << curp.X << ' ' << curp.Y << endl;
            gen_QImage_density(make_density(blocked_points.dim(),res),Qt::green).save("test.png");
        }
        count++;
        loc_val lval = gen_loc_val(curp,end,prevp,back2p);
        
        vector<Point> pathext = path_to_max(blocked_points,curp,lval);
        
        res.insert(res.end(),pathext.begin(),pathext.end()-1);
        
        back2p = prevp;
        prevp = curp;
        curp = pathext.back();
    }
    res.push_back(curp);
    return res;
}


vector<Point> rand_walk(const FArray2d<char> & blocked_points,Point begin, Point end){
    Point minedge = Point(0,0);
    Point maxedge = blocked_points.dim()-Point(1,1); 
    
    vector<Point> res;
    Point curp = begin;
    vector<Point> avaliable_pts;
    avaliable_pts.reserve(8);
    while(curp != end){
        res.push_back(curp);
        avaliable_pts.resize(0);
        iter_around(minedge,maxedge,curp,[&](Point p){
            if(!blocked_points[p]){
                avaliable_pts.push_back(p);
            }
        });
        Point randp = avaliable_pts[rand()%avaliable_pts.size()];
        curp = randp;
    }
    res.push_back(curp);
    return res;
}


vector<QPointF> conv_vec(const vector<Point>  & points){
    vector<QPointF> res(points.size());
    for(size_t i = 0; i < points.size(); i++){
        res[i] = middle_of(points[i]);
    }
    return res;
}
double connected_distance(const vector<QPointF> & points){
    double sum = 0;
    for(size_t i = 1; i < points.size(); i++){
        sum += distance(points[i-1],points[i]);
    }
    return sum;
}

void add_line(FArray2d<char> & maze,QPointF p1,QPointF p2){
    iter_points_between(p1,p2,[&](Point p){
        maze[p] = !maze[p];
    });
}


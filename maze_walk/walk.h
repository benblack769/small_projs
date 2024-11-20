#pragma once
#include <vector>
#include <cassert>
#include <iostream>
#include <QPointF>
#include "point.h"
#include "farray2d.h"
using namespace std;


double sqr(double x){
    return x*x;
}

double distance(QPointF p1,QPointF p2){
    return sqrt(sqr(p1.x()-p2.x()) + sqr(p1.y()-p2.y()));
}
QPoint q_pt(Point p){
    return QPoint(p.X,p.Y);
}
Point my_point(QPointF p){
    return Point{int(p.x()),int(p.y())};
}
Point point_dir(QPointF fdir){
    int xsign = fdir.x() > 0 ? 1 : -1;
    int ysign = fdir.y() > 0 ? 1 : -1;
    return Point(xsign,ysign);
}
double slope(QPointF s,QPointF e){
    double dx = e.x() - s.x();
    double dy = e.y() - s.y();
    return dx == 0 ? 10e6*dy : dy/dx;
}
double intercept(QPointF p,double slope){
    return p.y() - p.x() * slope;
}
struct line_int{
    qreal m;
    qreal b;
    
    qreal minx;
    qreal miny;
    qreal maxx;
    qreal maxy;
    line_int(QPointF s,QPointF e){
        m = slope(s,e);
        b = intercept(s,m);
        
        minx = min(s.x(),e.x());
        miny = min(s.y(),e.y());
        
        maxx = max(s.x(),e.x());
        maxy = max(s.y(),e.y());
    }
    bool inside_line_bounds(QPointF p)const{
        return p.x() > minx - 0.05 && p.x() < maxx + 0.05 &&
               p.y() > miny - 0.05 && p.y() < maxy + 0.05;
    }
};

QPointF intersect_point(const line_int & l1,const line_int & l2){
    qreal xint = (l2.b-l1.b)/(l1.m-l2.m);
    qreal yint = l1.m*xint+l1.b;
    QPointF intp(xint,yint);
    return intp;
}
bool in_line_bounds(const line_int & l1,const line_int & l2,QPointF intp){
    return l1.inside_line_bounds(intp) && l2.inside_line_bounds(intp);
}

QPointF middle_of(Point p){
    return QPointF(p.X+0.5,p.Y+0.5);
}
template<typename fn_ty>
void iter_points_between(QPointF start,QPointF end,fn_ty itfn){
    const Point endp = my_point(end);
    const Point tot_dir = point_dir(end-start);
    const Point xdir = Point(tot_dir.X,0);
    const Point ydir = Point(0,tot_dir.Y);
    const QPointF xdirf = q_pt(xdir);
    const QPointF ydirf = q_pt(ydir);
    const QPointF addvec(tot_dir.X*0.01,tot_dir.Y*0.01);
    const line_int moveline(start,end);
    
    QPointF curpf = start;
    while(my_point(curpf) != endp){
        QPointF bdrp = q_pt(my_point(curpf));
        
        itfn(my_point(curpf));
        
        const line_int edge1(bdrp,bdrp-xdirf);
        const line_int edge2(bdrp,bdrp-ydirf);
        QPointF int1 = intersect_point(moveline,edge1);
        QPointF int2 = intersect_point(moveline,edge2);
        if(in_line_bounds(moveline,edge1,int1)){
            curpf = int1 + addvec;
        }
        else if(in_line_bounds(moveline,edge2,int2)){
            curpf = int2 + addvec;
        }
        else{
            //cout << "Point failed: " << curpf.x() << " " << curpf.y() << endl;
            //cout << "end at: " << end.x() << " " << end.y() << endl;
            break;
            /*
            int sdalka = 102;
            sdalka *= 10;
            intersects(moveline, edge2);
            intersects(moveline, edge1);
            assert(false && "intersects with neither line");*/
        }
    }
    itfn(endp);
}
bool direct_path_clear(const FArray2d<char> & maze,QPointF p1,QPointF p2){
    bool path_clear = true;
    iter_points_between(p1,p2,[&](Point p){
        path_clear = path_clear && maze[p];
    });
    return path_clear;
}

#include "fields.h"
#include "populate.h"
#include <cstdlib>
#include <cmath>
#include <iostream>

const double STEP_SIZE = 0.001;
const int NUM_ITERS = 1000000;
double fperturb(){
  return rand()/double(RAND_MAX);
}
double sqr(double x){
  return x * x;
}
struct Point{
  double x,y;
  // void operator *=(Point p){
  //   x *= p.x;
  //   y *= p.y;
  // }
  void operator *=(double f){
    x *= f;
    y *= f;
  }
  void operator +=(Point p){
    x += p.x;
    y += p.y;
  }
  Point operator +(Point p){
    Point p2 = *this;
    p2 += p;
    return p2;
  }
  Point operator *(double p){
    Point p2 = *this;
    p2 *= p;
    return p2;
  }
  Point operator -(Point p){
    Point p2 = *this;
    p2 += -p;
    return p2;
  }
  Point operator -(){
    return Point{-x,-y};
  }
  double len(){
    return sqrt(sqr(x) + sqr(y));
  }
  void perturb(){
    x += fperturb()*(STEP_SIZE * 0.01f);
    y += fperturb()*(STEP_SIZE * 0.01f);
  }
};
using VecP = std::vector<Point>;

void iter(Point curp, Point & nextp, Point adj_p, double weight){
  Point vec = adj_p - curp;
  double dist = vec.len();
  nextp += -vec * (STEP_SIZE * weight / (dist * sqr(dist)));
}

void iter_close(Point curp, Point & nextp, Point adj_p, double weight){
  Point vec = adj_p - curp;
  double dist = vec.len();
  nextp += -vec * (STEP_SIZE * weight / (dist*dist*dist*dist));
}
void iter(VecP & points, VecP & fut_points, Field & f){
  //std::copy(fut_points.begin(), fut_points.end(), points.begin());
  for(size_t i = 0; i < points.size(); i++){
    Point & fut_p = fut_points[i];
    Point cur_p = points[i];
    fut_p = cur_p;
    if(i+1 < points.size()){
      iter(cur_p, fut_p, points[i+1], -100);
      iter_close(cur_p, fut_p, points[i+1], 200);
    }
    if(i >= 1){
      iter(cur_p, fut_p, points[i-1], -100);
      iter_close(cur_p, fut_p, points[i-1], 200);
    }
    for(FieldPair fp : f.energy_mat[f.sequence[i]]){
      if(fp.i == i)
        continue;
      //if (i == 0) std::cout << fp.weight<< "\n";
      iter(cur_p, fut_p, points[fp.i], fp.weight);
    }
  }
}

VecP plot(Field L){
  VecP curp(L.sequence.size());
  VecP nextp(L.sequence.size());
  for(int i = 0; i < curp.size(); i++){
    curp[i].x = i;
    curp[i].y = rand()/double(RAND_MAX);
  }

  for(int i = 0; i < NUM_ITERS; i++){
      iter(curp, nextp, L);
      curp.swap(nextp);
  }
  return curp;
}
void print_plot(VecP plot){
  using namespace std;
  cout << "x,y\n";
  for(Point p : plot){
    cout << p.x << "," << p.y << "\n";
  }
}

int main(){
  populate();
  using namespace std;
  Field I = letters['I'];
  print_plot(plot(I));
}

#include <iostream>
#include "/home/dimka/eigen-master/Eigen/Core"

typedef Eigen::MatrixXd Matrix;
typedef Eigen::VectorXd Vector;

 using namespace std;
 int main()
 {
   Matrix x = Matrix::Random(400, 100);
   cout << " Hello GitHub " << endl ;
   return 0;
 }
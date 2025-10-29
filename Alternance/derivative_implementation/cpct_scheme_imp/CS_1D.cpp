#include <iostream>
#include<cmath>
#include<vector>
#include"save_vector.hpp"
#include"CS_1D.hpp"

template <typename T>
T my_function(double x) {
  return std::sin(x) * std::cos(x);
}

int main(int argc,char* argv[]){
    std::size_t N=100;
    double L=2.0 * M_PI;
    double h= L/static_cast<double>(N);
    double alpha;
    int order = 2;
    //int derivative=1;

    if (argc > 1) N = static_cast<size_t>(std::stoul(argv[1]));
    //if (argc > 2) derivative = std::stoi(argv[2]);
    if (argc > 2) order = std::stoi(argv[2]);
    if (argc > 3) alpha = std::stoi(argv[3]); 
    if (argc > 4) {
        std::cerr << "Error: Too many arguments.\n";
        std::cerr << "Usage: " << argv[0] << " [number_of_points,] [alpha,] [order_of_precision,]\n";
     return 1;
  }

    std::cout << "N = " << N <<  ", order = " << order << std::endl;

    CS1D<double>* solver;

    if(order == 2){
        solver = new CS1D_o2<double>(h,N);}
    else if(order == 4){
        solver = new CS1D_o4<double>(h,N);}
    else if(order == 6){
        solver = new CS1D_o6<double>(h,N);}
    else if(order == 8){
        solver = new CS1D_o8<double>(h,N);}
    else {
    std::cerr << "Error: order must be 2 , 4,6 or 8 for a tridiagonal system.\n";
    return 1;
  }

  //solver -> run_algo(f);

  solver -> init_A_first_der();
  solver -> init_B_first_der();
  solver -> compute_rhs_first_der(my_function<double>);
  solver -> thomas_solver();

  auto d = solver->get_derivative();
  alpha = solver->get_alpha();
  auto beta = solver-> get_beta();
  auto a = solver->get_a();
  auto b = solver->get_b();
  auto c= solver->get_c();

  std::cout<< " your coefficients are : alpha = " <<alpha << ", beta = " <<beta<<", a = "<< a<<", b = "<<b<<", c = "<<c<<".\n";
  
  std::string filename = "approx_der_x.bin";
    
    
  if (save_vector_to_binary(filename, d )) {
      std::cout << "Vector successfully saved to " << filename << std::endl;
  } else {
      std::cerr << "Error saving vector to " << filename << std::endl;
  }

  delete solver;

  return 0;
 
}


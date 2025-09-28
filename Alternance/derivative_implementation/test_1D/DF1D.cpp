#include <iostream>
#include <cmath>
#include <vector>
#include "DF1D.hpp"
#include "save_vector.hpp"

template <typename T>
T function(const T& x){
  return std::sin(x);
}

template <typename T>
T deriv_exact( const T& x){
  return std::cos(x);
}

int main(int argc, char* argv[]){
  std::size_t N = 100;
  double L = 2.0 * M_PI;
  int order = 2;
  int derivative = 1;
  
  if (argc > 1) N = static_cast<size_t>(std::stoul(argv[1]));
  if (argc > 2) derivative = std::stoi(argv[2]);
  if (argc > 3) order = std::stoi(argv[3]);
  if (argc > 4) {
     std::cerr << "Error: Too many arguments.\n";
     std::cerr << "Usage: " << argv[0] << " [number_of_points,] [derivative,] [order_of_precision,]\n";
     return 1;
  }

  std::cout << "N = " << N << ", derivative = " << derivative << ", order = " << order << std::endl;

  double h = L / static_cast<double>(N);
  std::vector<double> exact_derivative(N);

  DF1D<double>* solver;
  
  if (order == 2){
    solver = new DF1D_o2<double>(h, N);}
  else if ( order == 4){
    solver = new DF1D_o4<double>(h, N);}
  else {
    std::cerr << "Error: order must be 2 or 4.\n";
    return 1;
  }

  solver->load_x();
  solver->load_f(function<double>);
  
    if (derivative == 1){
      solver->first_der(); }
    else if (derivative == 2){
      solver->second_der(); }

    auto d = solver->get_derivative();
    auto x = solver->get_x();

    /*std::cout<<"x= ";
    for(std::size_t i=0; i<N ; ++i){
    std::cout<<x[i]<<" ,";}*/ 

    for (std::size_t i=0; i<N; ++i){
      exact_derivative[i]=deriv_exact(x[i]);
    }

    std::string filename1 = "approx_der.bin";
    std::string filename2 = "exact_der.bin";
    
    if (save_vector_to_binary(filename1, d )) {
        std::cout << "Vector successfully saved to " << filename1 << std::endl;
    } else {
        std::cerr << "Error saving vector to " << filename1 << std::endl;
    }

    if (save_vector_to_binary(filename2, exact_derivative)) {
        std::cout << "Vector successfully saved to " << filename2 << std::endl;
    } else {
        std::cerr << "Error saving vector to " << filename2 << std::endl;
    }

    
    delete solver;

    return 0;
  
}

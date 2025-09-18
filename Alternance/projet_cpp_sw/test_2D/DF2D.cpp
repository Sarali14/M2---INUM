#include <iostream>
#include <cmath>
#include <vector>
#include "DF2D.hpp"
#include "save_vector.hpp"
#include "vector_2D.hpp"

template <typename T>
T my_function(double x, double y) {
  return std::sin(x) * std::cos(y);
}

int main(int argc, char* argv[]){
  std::size_t Nx = 100;
  std::size_t Ny = 100;
  double Lx = 2.0 * M_PI;
  double Ly = 2.0 * M_PI;
  int order = 2;
  int derivative = 1;
  
  if (argc > 1) Nx = static_cast<size_t>(std::stoul(argv[1]));
  if (argc > 2) Ny = static_cast<size_t>(std::stoul(argv[2]));
  if (argc > 3) derivative = std::stoi(argv[3]);
  if (argc > 4) order = std::stoi(argv[4]);
  if (argc > 5) {
     std::cerr << "Error: Too many arguments.\n";
     std::cerr << "Usage: " << argv[0] << " [number_of_points_x,] [number_of_points_y,] [derivative,] [order_of_precision,]\n";
     return 1;
  }

  std::cout << "Nx = " << Nx << "Ny = "<< Ny << ", derivative = " << derivative << ", order = " << order << std::endl;

  double h = Lx / static_cast<double>(Nx);
  double k = Ly / static_cast<double>(Ny); 

  DF2D<double>* solver;
  
  if (order == 2){
    solver = new DF2D_o2<double>(h, k, Nx, Ny );}
  else if ( order == 4){
    solver = new DF2D_o4<double>(h , k, Nx, Ny);}
  else {
    std::cerr << "Error: order must be 2 or 4.\n";
    return 1;
  }
  
  solver->load_f(my_function<double>);
  
    if (derivative == 1){
      solver->first_der_x();
	solver->first_der_y(); }
    else if (derivative == 2){
      solver->second_der_x();
	solver->second_der_y();
	solver->laplacien(derivative);}

    auto d_x = solver->get_derivative_x();
    auto d_y = solver->get_derivative_y();
    

    std::string filename1 = "approx_der_x.bin";
    std::string filename2 = "approx_der_y.bin";
    
    
    if (save_vector_to_binary(filename1, d_x )) {
        std::cout << "Vector successfully saved to " << filename1 << std::endl;
    } else {
        std::cerr << "Error saving vector to " << filename1 << std::endl;
    }

    if (save_vector_to_binary(filename2, d_y)) {
        std::cout << "Vector successfully saved to " << filename2 << std::endl;
    } else {
        std::cerr << "Error saving vector to " << filename2 << std::endl;
    }

        if (derivative == 2){
      auto delta = solver->get_laplacien();
      std::string filename3 = "approx_laplacien.bin";
      if (save_vector_to_binary(filename3, delta)) { 
        std::cout << "Vector successfully saved to " << filename3 << std::endl;
      } else {
        std::cerr << "Error saving vector to " << filename3 << std::endl;
      }
    }

    delete solver;

    return 0;
  
}


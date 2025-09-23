#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <variant>
#include <type_traits>
#include "derivative.hpp"
#include "save_vector.hpp"
#include "vector_3D.hpp"

template <typename T>
T my_function(double x, double y, double z) {
  return std::sin(x) * std::cos(y) * std::sin(z);
}

int main(int argc, char* argv[]){
  std::size_t Nx = 100;
  std::size_t Ny = 100;
  std::size_t Nz = 100;
  domain_size L = {2.0 * M_PI,2.0 * M_PI,2.0 * M_PI};
  std::string method = "DFo2" ;
  
  if (argc > 1) Nx = static_cast<size_t>(std::stoul(argv[1]));
  if (argc > 2) Ny = static_cast<size_t>(std::stoul(argv[2]));
  if (argc > 3) Nz = static_cast<size_t>(std::stoul(argv[3]));
  if (argc > 4) method = argv[4];
  if (argc > 5) {
     std::cerr << "Error: Too many arguments.\n";
     std::cerr << "Usage: " << argv[0] << " [number_of_points_x,] [number_of_points_y,] [derivative,] [order_of_precision,]\n";
     return 1;
  }

  std::cout << "Nx = " << Nx << "Ny = "<< Ny << "Nz = " << Nz << ", method = " << method
	    << std::endl;
  Grid3D g{Nx,Ny,Nz};
  step_size H={g,L};

  std::variant<std::monostate,DF_o2<double>, DF_o4<double>,
	       spectral<double>> solver_variant;
  
  if (method == "DFo2" ){
    solver_variant = DF_o2<double>(H, g );}
  else if ( method == "DFo4"){
    solver_variant =  DF_o4<double>(H , g);}
  else if (method == "spectral"){
    solver_variant =  spectral<double>(H, g, L, spectral<double>::Direction::x);}

    auto function_wrapper = [&](std::size_t i, std::size_t j, std::size_t k) {
      double x = i * H.hi;
      double y = j * H.hj;
      double z = k * H.hk;
      return my_function<double>(x, y, z);
  };


  std::visit([&](auto& solver) {
    if constexpr (!std::is_same_v<std::monostate, std::decay_t<decltype(solver)>>){
     solver.laplacien(function_wrapper); // Pass the correct wrapper function

      const auto& d_x = solver.get_derivative_x();
      const auto& d_y = solver.get_derivative_y();
      const auto& d_z = solver.get_derivative_z();
      const auto& delta = solver.get_laplacien();
      
      // Save the results
      save_vector_to_binary("approx_der_x.bin", d_x);
      save_vector_to_binary("approx_der_y.bin", d_y);
      save_vector_to_binary("approx_der_z.bin", d_z);
      save_vector_to_binary("approx_laplacien.bin", delta);

      std::cout << "Vectors successfully saved." << std::endl;}

  }, solver_variant); 
  /*

    auto d_x = solver->get_derivative_x();
    auto d_y = solver->get_derivative_y();
    auto d_z = solver->get_derivative_z();
    auto delta = solver->get_laplacien();
    

    std::string filename1 = "approx_der_x.bin";
    std::string filename2 = "approx_der_y.bin";
    std::string filename3 = "approx_der_z.bin";
    std::string filename4 = "approx_laplacien.bin";
    
    
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
    if (save_vector_to_binary(filename3, d_z)) {
        std::cout << "Vector successfully saved to " << filename3 << std::endl;
    } else {
        std::cerr << "Error saving vector to " << filename3 << std::endl;
    }
    if (save_vector_to_binary(filename4, delta)) {
	std::cout << "Vector successfully saved to " << filename4 << std::endl;
    } else {
	std::cerr << "Error saving vector to " << filename4 << std::endl;
    }


    delete solver;*/

    return 0;
  
}

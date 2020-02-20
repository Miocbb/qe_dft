#include "parser/OptionPrinter.hpp"
#include "index_matcher.h"
#include "matrix_io.h"

#include <Eigen/Dense>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <exception>
#include <iostream>
#include <memory>
#include <stdio.h>
#include <string>
#include <vector>


namespace po = boost::program_options;
using std::string;
using Matrix = Eigen::MatrixXd;
using SharedMatrix = std::shared_ptr<Matrix>;
using std::vector;

int main(int ac, char *av[])
{
    std::string appName = boost::filesystem::basename(av[0]);

    string file_S;        // AO overlap matrix file.
    string file_C_ref;    // CO coefficient matrix file for N-system.
    string file_C_remove; // CO coefficient matrix file for N-1 system.
    string index_options;

    po::options_description args("QE-DFT (quasi-particle energy from DFT) "
                                 "calculation for excited state problems.");
    args.add_options()("help,h", "Produce help massage.")(
        "index", po::value<string>(&index_options)->default_value("all"),
        "Indices that to find the corresponding orbitals. Comma separated. "
        "\"a-b\" means \"a, a+1, ..., b\".")(
        "file_CO_ref", po::value<string>(&file_C_ref)->required(),
        "Binary file generated from QM4D for CO coefficient matrix from "
        "N-electron system.")(
        "file_CO_remove", po::value<string>(&file_C_remove)->required(),
        "Binary file generated from QM4D for CO coefficient matrix from "
        "(N-1)-electron system.")(
        "file_S", po::value<string>(&file_S)->required(),
        "Binary file generated from QM4D for AO overlap matrix.");

    po::positional_options_description pst_args;
    pst_args.add("file_CO_ref", 1);
    pst_args.add("file_CO_remove", 1);
    pst_args.add("file_S", 1);

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(ac, av)
                      .options(args)
                      .positional(pst_args)
                      .run(),
                  vm);
        if (vm.count("help")) {
            // Use the code from the following source.
            // http://www.radmangames.com/programming/how-to-use-boost-program_options
            rad::OptionPrinter::printStandardAppDesc(appName, std::cout, args,
                                                     &pst_args);
            return 0;
        }

        po::notify(vm);
    } catch (...) {
        throw;
    }

    vector<SharedMatrix> S;     // AO overlap matrix.
    vector<SharedMatrix> C_N;   // CO coefficient matrix for N-system.
    vector<SharedMatrix> C_N_1; // CO coefficient matrix for N-1 system.

    try {
        S = read_matrices_from_txt(file_S);
        C_N = read_matrices_from_txt(file_C_ref);
        C_N_1 = read_matrices_from_txt(file_C_remove);
    } catch (...) {
        std::cout << "Overlap matrix binary file: " << file_S << std::endl;
        std::cout << "N system CO coefficient matrix binary file: "
                  << file_C_ref << std::endl;
        std::cout << "N-1 system CO coefficient matrix binary file: "
                  << file_C_remove << std::endl;
        throw std::runtime_error("Error: failed to load data from the input "
                                 "data.");
    }

    if (S.size() != 1) {
        throw "Error: no AO overlap matrix data.\n";
    }
    if (C_N.size() != 2) {
        throw "Error: miss one spin for CO coefficient matrix for N system\n";
    }
    if (C_N_1.size() != 2) {
        throw "Error: miss one spin for CO coefficient matrix for N-1 system\n";
    }
    // Coefficient matrix generated from QM4D needs to be transposed.
    for (size_t i = 0; i < 2; ++i) {
        C_N[i]->transposeInPlace();
        C_N_1[i]->transposeInPlace();
    }


#ifdef DEBUG_PRINT
    std::cout << "Overlap matrix:\n" << *S[0] << std::endl;
    std::cout << "CO coef matrix N system: spin=0\n" << *C_N[0] << std::endl;
    std::cout << "CO coef matrix N system: spin=1\n" << *C_N[1] << std::endl;
    std::cout << "CO coef matrix N-1 system: spin=0\n" << *C_N_1[0] << std::endl;
    std::cout << "CO coef matrix N-1 system: spin=1\n" << *C_N_1[1] << std::endl;
#endif

    qe_dft::Qedft orbitals(S[0], C_N, C_N_1);
    std::vector<size_t> indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto index_pair = orbitals.get_corresponding_orbitals(indices);
    for (size_t i = 0; i < indices.size(); ++i) {
        printf("Input index=%zu, find index: alpha=%zu, beta=%zu\n", indices[i],
               index_pair[i].first, index_pair[i].second);
    }
}

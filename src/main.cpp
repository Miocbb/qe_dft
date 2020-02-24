#include "matrix_io.h"
#include "parser/OptionPrinter.hpp"
#include "qe_dft.h"

#include <Eigen/Dense>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <exception>
#include <iostream>
#include <memory>
#include <stdio.h>
#include <string>
#include <unordered_set>
#include <vector>

using std::string;
using Matrix = Eigen::MatrixXd;
using SharedMatrix = std::shared_ptr<Matrix>;
using std::vector;

static const double au2eV = 27.211324570273;

namespace {
namespace po = boost::program_options;

/**
 * @brief remove duplicated elements in the vector in place.
 */
void remove_duplicates(vector<size_t> &in)
{
    std::unordered_set<size_t> s(in.begin(), in.end());
    in.assign(s.begin(), s.end());
}

/**
 * @brief split string with given delimeter.
 * @param [in] str: input string to be splitted.
 * @param [in] delim: delimeter. Default `delim = ','`.
 * @return vector<string>: the splitted string.
 */
vector<string> str_split(const string &str, const char delim = ',')
{
    vector<string> str_split;
    std::stringstream str_stream(str + std::string(1, delim));
    std::string word;
    while (std::getline(str_stream, word, delim)) {
        str_split.push_back(word);
    }
    return str_split;
}

bool str_find(const string &str, const string &sub_str)
{
    size_t found = str.find(sub_str);
    return found != std::string::npos;
}

/**
 * @brief Parse the string expression of excitation to give all the orbital
 * indices.
 *
 * @param [in] index_expr: excitation indices expression. Two type of
 * expressions are supported. (1). "n" means the HOMO->HOMO+n excitation; (2)
 * "n1-n2" means all the excitation from HOMO->HOMO+n1 to HOMO->HOMO+n2. Each
 * expression is seperated by comma. You can make "1, 2, 2-6", which will be
 * understand as "1, 2, 3, 4, 5, 6".
 * @param [in] nocc: number of occupied orbitals.
 *
 * @return vector<size_t>: the orbital indices (starting from zero) related to
 * the expressed excitations in an increasing order. The duplicated indices are
 * removed.
 */
vector<size_t> parse_exci_index(const string &index_expr, size_t nocc)
{
    vector<string> index_expr_split = str_split(index_expr, ',');
    vector<size_t> rst;
    for (string idx_i : index_expr_split) {
        if (str_find(idx_i, "-")) {
            vector<string> range;
            size_t begin = 0;
            size_t end = 0;
            try {
                range = str_split(idx_i, '-');
                begin = std::stoi(range[0]);
                end = std::stoi(range[1]);
            } catch (...) {
                throw std::runtime_error("Cannot parse excitation index "
                                         "expression: whole expression=" +
                                         index_expr + ". Failed at: " + idx_i);
            }
            if (begin > end || begin <= 0) {
                throw std::runtime_error(
                    "Invalid excitation index expression: whole expression=" +
                    index_expr + ". Failed at: " + idx_i);
            }
            for (size_t i = begin; i <= end; ++i) {
                rst.push_back(i);
            }
        } else {
            size_t idx = 0;
            try {
                idx = std::stoi(idx_i);
            } catch (const std::exception &) {
                throw std::runtime_error("Cannot parse excitation index "
                                         "expression: whole expression=" +
                                         index_expr + ". Failed at: " + idx_i);
            }
            if (idx <= 0) {
                throw std::runtime_error(
                    "Invalid excitation index expression: whole expression=" +
                    index_expr + ". Failed at: " + idx_i);
            }
            rst.push_back(std::stoi(idx_i));
        }
    }
    remove_duplicates(rst);
    std::sort(rst.begin(), rst.end());
    for (size_t &i : rst) {
        i = nocc - 1 + i;
    }
    return rst;
}

po::variables_map parse_args(int ac, char *av[])
{
    std::string appName = boost::filesystem::basename(av[0]);
    po::options_description args("QE-DFT (quasi-particle energy from DFT) "
                                 "calculation for excited state problems.");
    args.add_options()("help,h", "Produce help massage.")(
        "index", po::value<string>()->default_value("1"),
        "Excitation indices expression. Two type of expressions are "
        "supported.(1). \"n\" means the HOMO->HOMO+n excitation; (2) \"n1-n2\" "
        "means all the excitation from HOMO->HOMO+n1 to HOMO->HOMO+n2. Each "
        "expression is ONLY seperated by comma.You can make \"1,2,2-6\", which "
        "will "
        "be understand as \"1,2,3,4,5,6\".")(
        "nocc", po::value<size_t>()->required(),
        "number of occupied orbital.")(
        "file_CO_N", po::value<string>()->required(),
        "Binary file generated from QM4D for CO coefficient matrix from "
        "N-electron system.")(
        "file_CO_Nrm", po::value<string>()->required(),
        "Binary file generated from QM4D for CO coefficient matrix from "
        "(N-1)-electron system.")(
        "file_S", po::value<string>()->required(),
        "Binary file generated from QM4D for AO overlap matrix.")(
        "file_eig_Nrm", po::value<string>()->required(),
        "Binary file generated from QM4D for orbital energy matrix.");

    po::positional_options_description pst_args;
    pst_args.add("nocc", 1);
    pst_args.add("file_CO_N", 1);
    pst_args.add("file_CO_Nrm", 1);
    pst_args.add("file_S", 1);
    pst_args.add("file_eig_Nrm", 1);

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
    return vm;
}

} // local namespace

int main(int ac, char *av[])
{
    // parse arguments.
    auto vm = parse_args(ac, av);

    // load data
    string file_S = vm["file_S"].as<string>();
    string file_C_ref = vm["file_CO_N"].as<string>();
    string file_C_remove  = vm["file_CO_Nrm"].as<string>();
    string file_OrbE_N_1 = vm["file_eig_Nrm"].as<string>();
    string index_options = vm["index"].as<string>();
    const size_t nocc = vm["nocc"].as<size_t>();

    vector<size_t> indices = parse_exci_index(index_options, nocc);
    std::cout << "Excited orbitals: \n";
    for (size_t i = 0; i < indices.size(); ++i) {
        std::cout << indices[i] + 1 << ",";
    }
    std::cout << "\n";

    SharedMatrix S;             // AO overlap matrix.
    vector<SharedMatrix> C_N;   // CO coefficient matrix for N-system.
    vector<SharedMatrix> C_N_1; // CO coefficient matrix for N-1 system.
    SharedMatrix orbE_N_1;
    try {
        S = read_matrices_from_txt(file_S)[0];
        C_N = read_matrices_from_txt(file_C_ref);
        C_N_1 = read_matrices_from_txt(file_C_remove);
        orbE_N_1 = read_matrices_from_txt(file_OrbE_N_1)[0];
    } catch (...) {
        std::cout << "Overlap matrix binary file: " << file_S << std::endl;
        std::cout << "N system CO coefficient matrix binary file: "
                  << file_C_ref << std::endl;
        std::cout << "N-1 system CO coefficient matrix binary file: "
                  << file_C_remove << std::endl;
        std::cout << "N-1 system orbital energy binary file: " << file_OrbE_N_1
                  << std::endl;
        throw std::runtime_error("Error: failed to load data from the input "
                                 "data.");
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
    std::cout << "Overlap matrix:\n" << *S << std::endl;
    std::cout << "CO coef matrix N system: spin=0\n" << *C_N[0] << std::endl;
    std::cout << "CO coef matrix N system: spin=1\n" << *C_N[1] << std::endl;
    std::cout << "CO coef matrix N-1 system: spin=0\n"
              << *C_N_1[0] << std::endl;
    std::cout << "CO coef matrix N-1 system: spin=1\n"
              << *C_N_1[1] << std::endl;
    std::cout << "Orbital energy matrix for N-1 system: \n"
              << *orbE_N_1 << std::endl;
#endif

    // do QE-DFT calculation.
    qe_dft::Qedft orbitals(nocc, S, C_N, C_N_1, orbE_N_1);
    // find corrsponding indices in N-1 system for the N system excitation.
    auto matched_index_pair = orbitals.get_corresponding_orbitals(indices);
    for (size_t i = 0; i < indices.size(); ++i) {
        printf("Input index=%zu, find index: alpha=%zu, beta=%zu\n", indices[i],
               matched_index_pair[i].first, matched_index_pair[i].second);
    }
    // calculate excitation energy with the corresponding indices.
    auto exciE_pair = orbitals.excitation_energies(indices);
    for (size_t i = 0; i < indices.size(); ++i) {
        printf("Excitation: %zu->%zu. Used orbital: %zu->(alpha=%zu, "
               "beta=%zu). Singlet = %.3f (eV), Triplet = %.3f (eV)\n",
               nocc, indices[i] + 1, nocc,
               matched_index_pair[i].first + 1,
               matched_index_pair[i].second + 1, exciE_pair[i].first * au2eV,
               exciE_pair[i].second * au2eV);
    }
}

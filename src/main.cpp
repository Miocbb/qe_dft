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
 * @param [in] nocc_a: number of alpha occupied orbitals in the N system.
 *
 * @return vector<size_t>: the orbital indices (starting from zero) related to
 * the expressed excitations in an increasing order. The duplicated indices are
 * removed.
 */
vector<size_t> parse_exci_index(const string &index_expr, size_t nocc_a)
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
        i = nocc_a - 1 + i;
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
        "will be understand as \"1,2,3,4,5,6\".")(
        "overlap_check", po::value<bool>()->default_value(true),
        "If or not doing overlap check to selecting orbitals. Default=true.")(
        "nocc_a", po::value<size_t>()->required(),
        "number of alpha occupied orbital for the N system.")(
        "nocc_b", po::value<size_t>()->required(),
        "number of beta occupied orbital for the N system.")(
        "rm_or_add", po::value<string>()->required(),
        "Doing N-1 or N+1 QE-DFT calculation. Only choices are \"N-1\" or "
        "\"N+1\".")(
        "file_CO_N", po::value<string>()->required(),
        "Text file generated from QM4D for CO coefficient matrix from "
        "N-electron system.")(
        "file_CO_Nrm", po::value<string>()->required(),
        "Text file generated from QM4D for CO coefficient matrix from "
        "(N-1)-electron system.")(
        "file_S", po::value<string>()->required(),
        "Text file generated from QM4D for AO overlap matrix.")(
        "file_eig_Nrm", po::value<string>()->required(),
        "Text file generated from QM4D for orbital energy matrix.");

    po::positional_options_description pst_args;
    pst_args.add("nocc_a", 1);
    pst_args.add("nocc_b", 1);
    pst_args.add("rm_or_add", 1);
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
            std::exit(EXIT_SUCCESS);
        }

        const string rm_or_add = vm["rm_or_add"].as<string>();
        if (rm_or_add != "N-1" && rm_or_add != "N+1") {
            throw std::runtime_error(
                "Error to set N-1 or N+1 QE-DFT calculation. Supported setting "
                "is \"N-1\" or \"N+1\".");
        }

        po::notify(vm);
    } catch (...) {
        throw;
    }

    return vm;
}

} // namespace

int main(int ac, char *av[])
{
    // parse arguments.
    auto vm = parse_args(ac, av);

    // load data
    string file_S = vm["file_S"].as<string>();
    string file_C_ref = vm["file_CO_N"].as<string>();
    string file_C_remove = vm["file_CO_Nrm"].as<string>();
    string file_OrbE_N_1 = vm["file_eig_Nrm"].as<string>();
    string index_options = vm["index"].as<string>();
    const vector<size_t> nocc = {vm["nocc_a"].as<size_t>(),
                                 vm["nocc_b"].as<size_t>()};
    // alpha electron number should be greater or equal to beta electron number.
    if (nocc[0] < nocc[1]) {
        throw std::runtime_error(
            "Error: alpha electron number should be greater or equal to beta "
            "electron number! You need to modify your all input matrix data to "
            "meet this criteria!");
    }

    vector<size_t> indices = parse_exci_index(index_options, nocc[0]);
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

    if (vm["rm_or_add"].as<string>() == "N-1") {
        bool is_open_shell = (nocc[0] == nocc[1]);
        if (is_open_shell) {
            // open-shell N-1 system.
            qe_dft::QedftRemovalOpenShell qedft(nocc, S, C_N, C_N_1, orbE_N_1);
            for (size_t exci_idx : indices) {
                size_t gs_orb_b = qedft.gs_orbital_index();

                // do QE-DFT without overlap checking.
                double E_s_nocheck =
                    qedft.get_singlet_excitation(gs_orb_b, exci_idx, exci_idx);
                double E_t_nocheck =
                    qedft.get_triplet_excitation(gs_orb_b, exci_idx);
                printf("Excitation without orbital checking: %zu->%zu. Singlet "
                       "= %.3f (eV), Triplet = %.3f (eV)\n",
                       gs_orb_b + 1, exci_idx + 1, E_s_nocheck * au2eV,
                       E_t_nocheck * au2eV);

                // do QE-DFT with overlap checking.
                if (vm["overlap_check"].as<bool>()) {
                    auto used_gs_orb_b =
                        qedft.get_corresponding_orbital(qe_dft::Beta, gs_orb_b);
                    auto used_exci_orb_a = qedft.get_corresponding_orbital(
                        qe_dft::Alpha, exci_idx);
                    auto used_exci_orb_b =
                        qedft.get_corresponding_orbital(qe_dft::Beta, exci_idx);
                    double E_s = qedft.get_singlet_excitation(
                        used_gs_orb_b.first, used_exci_orb_a.first,
                        used_exci_orb_b.first);
                    double E_t = qedft.get_triplet_excitation(
                        used_gs_orb_b.first, used_exci_orb_a.first);
                    printf(
                        "Excitation with orbital checking: %zu->%zu. Singlet = "
                        "%.3f (eV), Triplet = %.3f (eV)\n",
                        gs_orb_b + 1, exci_idx + 1, E_s * au2eV, E_t * au2eV);
                    printf("Used orbital to ground state:  spin = beta,  idx = "
                           "%3zu, overlap = %.3f\n",
                           used_gs_orb_b.first + 1, used_gs_orb_b.second);
                    printf("Used orbital to excited state: spin = alpha, idx = "
                           "%3zu, overlap = %.3f\n",
                           used_exci_orb_a.first + 1, used_exci_orb_a.second);
                    printf("Used orbital to excited state: spin = beta,  idx = "
                           "%3zu, overlap = %.3f\n",
                           used_exci_orb_b.first + 1, used_exci_orb_b.second);
                }
                printf("\n");
            }
        } else {
            // close-shell N-1 system.
            qe_dft::QedftRemovalCloseShell qedft(nocc, S, C_N, C_N_1, orbE_N_1);
            for (size_t exci_idx : indices) {
                size_t gs_orb_a = qedft.gs_orbital_index();

                // do QE-DFT without overlap checking.
                double E_d_nocheck =
                    qedft.get_doublet_excitation(gs_orb_a, exci_idx);
                printf("Excitation without orbital checking: %zu->%zu. Doublet "
                       "= %.3f (eV)\n",
                       gs_orb_a + 1, exci_idx + 1, E_d_nocheck * au2eV);

                if (vm["overlap_check"].as<bool>()) {
                    auto used_gs_orb_a = qedft.get_corresponding_orbital(
                        qe_dft::Alpha, gs_orb_a);
                    auto used_exci_orb_a = qedft.get_corresponding_orbital(
                        qe_dft::Alpha, exci_idx);
                    double E_d = qedft.get_doublet_excitation(
                        used_gs_orb_a.first, used_exci_orb_a.first);
                    printf("Excitation: %zu->%zu. Doublet = %.3f (eV)\n",
                           gs_orb_a + 1, exci_idx + 1, E_d * au2eV);
                    printf("Used orbital to ground state:  spin = alpha, idx = "
                           "%3zu, overlap = %.3f\n",
                           used_gs_orb_a.first + 1, used_gs_orb_a.second);
                    printf("Used orbital to excited state: spin = alpha, idx = "
                           "%3zu, overlap = %.3f\n",
                           used_exci_orb_a.first + 1, used_exci_orb_a.second);
                }
                printf("\n");
            }
        }
    } else {
        throw std::runtime_error("N+1 QE-DFT is not implemented.\n");
    }
}

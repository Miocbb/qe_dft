#include "OptionPrinter.hpp"

#include "boost/program_options.hpp"
#include "boost/filesystem.hpp"

#include <iostream>
#include <string>

namespace
{
  const size_t ERROR_IN_COMMAND_LINE = 1;
  const size_t SUCCESS = 0;
  const size_t ERROR_UNHANDLED_EXCEPTION = 2;

} // namespace

//-----------------------------------------------------------------------------
int t(int argc, char** argv)
{
  try
  {
    std::string appName = boost::filesystem::basename(argv[0]);
    int add = 0;
    int like = 0;
    std::vector<std::string> sentence;

    /** Define and parse the program options
     */
    namespace po = boost::program_options;
    po::options_description desc("Options");
    desc.add_options()
      ("help,h", "Print help messages")
      ("verbose,v", "print words with verbosity")
      ("word,w", po::value<std::vector<std::string> >(&sentence),
       "words for the sentence, specify multiple times")
      (",t", "just a temp option that does very little")
      ("necessary,n", po::value<std::string>()->required(), "give me anything")
      ("manual,m", po::value<std::string>(), "extract value manually")
      ("add", po::value<int>(&add)->required(), "additional options")
      ("like", po::value<int>(&like)->required(), "this");

    po::positional_options_description positionalOptions;
    positionalOptions.add("add", 1);
    positionalOptions.add("like", 1);

    po::variables_map vm;

    try
    {
      po::store(po::command_line_parser(argc, argv).options(desc)
                  .positional(positionalOptions).run(),
                vm); // throws on error

      /** --help option
       */
      if ( vm.count("help")  )
      {
        std::cout << "This is just a template app that should be modified"
                  << " and added to in order to create a useful command line"
                  << " application" << std::endl << std::endl;
        rad::OptionPrinter::printStandardAppDesc(appName,
                                                 std::cout,
                                                 desc,
                                                 &positionalOptions);
        return SUCCESS;
      }

      po::notify(vm); // throws on error, so do after help in case
                      // there are any problems
    }
    catch(boost::program_options::required_option& e)
    {
      rad::OptionPrinter::formatRequiredOptionError(e);
      std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
      rad::OptionPrinter::printStandardAppDesc(appName,
                                               std::cout,
                                               desc,
                                               &positionalOptions);
      return ERROR_IN_COMMAND_LINE;
    }
    catch(boost::program_options::error& e)
    {
      std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
      rad::OptionPrinter::printStandardAppDesc(appName,
                                               std::cout,
                                               desc,
                                               &positionalOptions);
      return ERROR_IN_COMMAND_LINE;
    }

    // can do this without fear because it is required to be present
    std::cout << "Necessary = "
              << vm["necessary"].as<std::string>() << std::endl;

    if ( vm.count("verbose") )
    {
      std::cout << "VERBOSE PRINTING" << std::endl;
    }
    if (vm.count("verbose") && vm.count("t"))
    {
      std::cout << "heeeeyahhhhh" << std::endl;
    }

    std::cout << "Required Positional, add: " << add
              << " like: " << like << std::endl;

    if ( sentence.size() > 0 )
    {
      std::cout << "The specified words: ";
      std::string separator = " ";
      if (vm.count("verbose"))
      {
        separator = "__**__";
      }
      for(size_t i=0; i<sentence.size(); ++i)
      {
        std::cout << sentence[i] << separator;
      }
      std::cout << std::endl;

    }

    if ( vm.count("manual") )
    {
      std::cout << "Manually extracted value: "
                << vm["manual"].as<std::string>() << std::endl;
    }

  }
  catch(std::exception& e)
  {
    std::cerr << "Unhandled Exception reached the top of main: "
              << e.what() << ", application will now exit" << std::endl;
    return ERROR_UNHANDLED_EXCEPTION;

  }

  return SUCCESS;

} // main

//-----------------------------------------------------------------------------

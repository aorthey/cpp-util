#include <iostream>
#include <sstream>
#include <string>

#define CUR_LOCATION "@" << __FILE__ << ":" << __FUNCTION__ << ":" << __LINE__ << ">>"
#define PRINT(msg) std::cout << CUR_LOCATION << " >> " << msg << std::endl
#define ABORT(msg) PRINT(msg); throw msg;
#define EXIT(msg) PRINT(msg); exit;
namespace util{

	class MyStream: public std::ostream
	{
	    // Write a stream buffer that prefixes each line with Plop
	    class MyStreamBuf: public std::stringbuf
	    {
		std::ostream&   output;
		public:
		    MyStreamBuf(std::ostream& str)
			:output(str)
		    {}

		// When we sync the stream with the output. 
		// 1) Output Plop then the buffer
		// 2) Reset the buffer
		// 3) flush the actual output stream we are using.
		virtual int sync ( )
		{
			output <<  " " << str();
		    str("");
		    output.flush();
		    return 0;
		}
	    };

		MyStreamBuf buffer;
		std::string prefix;
	    public:
		MyStream(std::ostream& str): buffer(str),std::ostream(&buffer)
		{
		}
	};

	MyStream stream(std::cout);
	#define cout stream << CUR_LOCATION
}

int main()
{
	util::cout << 1 << 2 << 3 << std::endl << 5 << 6 << std::endl << 7 << 8 << std::endl;
	util::cout << 1 << 2 << 3 << std::endl << 5 << 6 << std::endl << 7 << 8 << std::endl;
	return 0;
}


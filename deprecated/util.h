#pragma once
#include <vector>
#include <cassert>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <sstream>

//This is an experimental testbed for several new programming paradigms
// and some proof of concepts. this includes:
//
// - variadic templates
// - overloaded ostream
// - own graph implementation
// - own vector/matrix implemenations by overloading std::vector
//
// TODO:
//
// - move bigger chunks to own files -> util_stream.cpp, util_graph.cpp,
// util_datastructures.cpp
// - revise graph structure
// - implement backward/forward value iteration for graphs
// - PrefixStream -> is there a more efficient solution?

namespace util{

//----------------------------------------------
//cout stream implementation -> util_stream.cpp ?
//----------------------------------------------

//-- PrefixStream: 
// \descr Extend the ostream object by adding a prefix to each outputline
// Example: if we call a cout message in a function called test, in a file
// called environment, on line number 17, then our output would generate
// environment_:test__:17_>> message
//
// We also included a message counter, which will show the current message
// number. 
//
// Every other behaviour should imitate cout
class PrefixStream: public std::ostream
{
	public:
		static std::string func;
		static std::string file;
		static int line;
		static uint msg_counter;
    class PrefixStreamStringBuffer: public std::stringbuf{
        std::ostream& output;
        public:
            PrefixStreamStringBuffer(std::ostream& str):output(str){};

				//sync is called everytime when the stream is flushed
        virtual int sync(){
 					//prefix all stdout message with
 					//NUMBER,FILENAME,FUNCTIONNAME,LINENUMBER
					output << "[" 
						<< std::setfill('_') << std::left << std::setw(3) << msg_counter++ << "]" 
						<< std::setfill('_') << std::left << std::setw(10) << file<<  ":" 
						<< std::setfill('_') << std::left << std::setw(10) << func<< ":" 
						<< std::setfill('_') << std::left << std::setw(4) << line << ">> " 
						<< str();
					str("");
					output.flush();
					return 0;
        };
    };

    PrefixStreamStringBuffer buffer;
    public: PrefixStream(std::ostream& str):std::ostream(&buffer),buffer(str){};
};

uint PrefixStream::msg_counter = 0;
std::string PrefixStream::func = "none";
std::string PrefixStream::file = "none";
int PrefixStream::line = -1;

PrefixStream sout(std::cout);


//helper class to set the neccessary variables in the stream
// and control what should happen with specific stream objects
// like std::endl
class Stream_Interface
{
	public: 
		Stream_Interface(	const std::string &funcName, 
										const std::string &fileName, 
										const int &line)
		{ 
				sout.func = funcName;
				sout.file = fileName;
				sout.line = line;
		};

		//here we just forward all objects to the stream operator 
		template <class T> Stream_Interface &operator<<(const T &v){ 
			sout << v; return *this; 
		};

		//change here the behaviour for stream objects like std::endl;
		Stream_Interface& operator<<( std::ostream&(*f)(std::ostream&) ){ 
			sout << f;
			return *this; 
		};


		~Stream_Interface() { 
				//here we could add a newline if neccessary
				//sout << std::endl;
		};
};

#define cout Stream_Interface(__FUNCTION__,__FILE__,__LINE__)


//----------------------------------------------
//Matrix, Vector implemenatations -> util_datastructures.cpp ?
//----------------------------------------------

template<class T>
class Matrix: public std::vector<T>{
  public:
    Matrix();
    explicit Matrix(int i, int j): std::vector<T>(i*j){d1=i;d2=j;dim=2;};

		T &operator()(int i, int j);
	  Matrix<T>& operator=(T rhs);
	  Matrix<T>& operator=(Matrix<T> &rhs);
		void eye(int i);
		T sum(void);
		T prod(void);
		void print(void){
			typename std::vector<T>::const_iterator it;
			unsigned int c = 0;
			for(it=this->begin();it!=this->end();it++){
				cout << *it << " ";
				if(c%this->d2 == this->d2-1 && c < this->d2*this->d1-1){
					cout << std::endl;
				}else{
				}
				c++;
			}
		};

		friend std::ostream &operator<<(std::ostream &os, const Matrix<T> &m){
			typename std::vector<T>::const_iterator it;
			int c = 0;
			os << "[";
			for(it=m.begin();it!=m.end();it++){
				os << *it;
				if(m.d1>1 && c%m.d2 == m.d2-1 && c < m.d2*m.d1-1){
					os << ";";
				}else{
					os << " ";
				}
				c++;
			}
			os << "]";
			return os;
		}; 
	private:
    uint dim;
    int d1;
    int d2;
};

template<class T>
class Vector: public Matrix<T>{
public:	
  Vector(): Matrix<T>(){};
  Vector(int i): Matrix<T>(1,i){};
};


//----------------------------------------------
//Vertex, Edge, Graph -> util_graph.cpp
//----------------------------------------------

namespace graph{
	using std::string;
	typedef struct Vertex{
		static unsigned int totalVertices;
		unsigned int id;
		string name;
		Vertex(const char *name){
			this->name.assign(name);
			this->id = totalVertices++;
		}
		~Vertex(){};
		friend std::ostream &operator<<(std::ostream &os, const Vertex &v){
			os << "(" << v.name << "[" << v.id << "])";
			return os;
		}; 
	}V;
	unsigned int Vertex::totalVertices = 0;

	typedef struct Edge{
		V *v;
		V *w;
		double cost;
		Edge(V &v, V &w, double cost = 1.0){
			this->v = &v;
			this->w = &w;
			this->cost = cost;
		}
		~Edge(){};
		friend std::ostream &operator<<(std::ostream &os, const Edge &e){
			os << *e.v << "-" << e.cost << "-" << *e.w;
			return os;
		}; 
	}E;

	class Graph{
		public:
		Vector<V> v;
		Vector<E> e;
		string name;
		Graph(const char *name = "default"){
			this->name.assign(name);
		}
		~Graph(){};
		void add(V &v){ this->v.push_back(v); };
		void setV(Vector<V> &vv){
			this->v = vv;
		}

		friend std::ostream &operator<<(std::ostream &os, const Graph &g){
			os << "---- Graph " << g.name << " ------" << std::endl;

			Vector<V>::const_iterator it;
			os << "Vertices: " << std::endl;
			for(it = g.v.begin();it!=g.v.end();it++){
				os << *it;
			}
			
			os << std::endl;
			Vector<E>::const_iterator eit;
			os << " Edges: " << std::endl;
			for(eit = g.e.begin();eit!=g.e.end();eit++){
					os << *eit << std::endl;
			}
			return os;
		}; 
		void backwardIteration( V v ){
			


		}


	}G;
};//namespace graph


#ifdef __GXX_EXPERIMENTAL_CXX0X__
	//variadic template implementation for unlimited element constructors
	template<typename T, typename ...A> Vector<T> VEC( A... args );
	template<typename T, typename ...A> Vector<T> SVEC( A... args );
	template<typename ...A> graph::Graph GRAPH( A... args );


	//GRAPH implementation: GRAPH( Vertex, Vertex, ...)
  void ghelper(graph::Graph &g, graph::Vertex &v){
  	g.add(v);
	}
	template<typename ...A>
  void ghelper(graph::Graph &g, graph::Vertex &v, A... args){
		g.add(v);
    if(sizeof...(args)) ghelper(g, args...);
  }

	template<typename ...A>
  graph::Graph GRAPH( A... args) {
      graph::Graph g;
      ghelper(g, args...);
      return g;
  }

	template<typename T>
  void vhelper(std::vector<T> &v, T &e){
  	v.push_back(e);
	}
	template<typename T, typename ...A>
  void vhelper(std::vector<T> &v, T &e, A... args){
  	v.push_back(e);
    if(sizeof...(args)) vhelper<T>(v, args...);
  }

	template<typename T, typename ...A>
  std::vector<T> SVEC( A... args ) {
		std::vector<T> v;
		vhelper<T>(v, args...);
		return v;
  }

	template<typename T>
  void vhelper(Vector<T> &v, T &e){
  	v.push_back(e);
	}
	template<typename T, typename ...A>
  void vhelper(Vector<T> &v, T &e, A... args){
  	v.push_back(e);
    if(sizeof...(args)) vhelper<T>(v, args...);
  }

	template<typename T, typename ...A>
  Vector<T> VEC( A... args ) {
		Vector<T> v;
		vhelper<T>(v, args...);
		return v;
	}
#endif

};//end namespace util
#include "util.cpp"

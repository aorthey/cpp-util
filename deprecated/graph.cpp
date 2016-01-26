#include <vector>
#include <string>
#include <stdio.h>

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
	void print(){ printf("(%s[%d])", name.c_str(), id); };
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
	void print(){ v->print(); printf(" - "); w->print(); printf("\n");};
}E;

typedef struct Graph{
	std::vector<V> v;
	std::vector<E> e;
	string name;
	Graph(const char *name = "default"){
		this->name.assign(name);
	}
	Graph(V vv[], const char *name = "default"){
		
		for(int i=0;i< sizeof(vv)/sizeof(V);i++){
		this->v.push_back(vv[i]);
		}
	}
	~Graph(){};
	void add(V &v){ this->v.push_back(v); };

	void print(){ 
		printf("GRAPH STRUCTURE %s\n", name.c_str());
		for(int i=0;i<v.size();i++){
			v.at(i).print();
		}
		printf("\n");
		for(int i=0;i<e.size();i++){
			e.at(i).print();
		}
	}
	void backwardIteration( V &v ){



	}
}G;

#ifdef __GXX_EXPERIMENTAL_CXX0X__
  void ghelper(Graph &g, Vertex &v){
  	g.add(v);
	}
	template<typename ...A>
  void ghelper(Graph &g, Vertex &v, A... args){
		g.add(v);
    if(sizeof...(args)) ghelper(g, args...);
  }

	template<typename ...A>
  Graph GRAPH( A... args ) {
      Graph g;
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
  std::vector<T> VEC( A... args ) {
		std::vector<T> v;
		vhelper<T>(v, args...);
		return v;
  }
#endif

/*
int main(){
	//Graph g = GRAPH( V(1,"a"), V(2,"b"), V(3,"c"), V(4,"d"), V(5,"e") );
	Graph g;
	g.v = VEC<Vertex>( V("a"), V("b"), V("c"), V("d"), V("e") );
	g.e = VEC<Edge>( 	E( g.v[0], g.v[1], 2 ), 
										E( g.v[0], g.v[0], 2 ), 
										E( g.v[1], g.v[2], 1 ), 
										E( g.v[1], g.v[3], 4 ), 
										E( g.v[2], g.v[3], 1 ), 
										E( g.v[2], g.v[0], 1 ), 
										E( g.v[3], g.v[2], 1 ), 
										E( g.v[3], g.v[4], 1 ) );


	g.print();
	g.backwardIteration( V("a") );
	


	return 0;
}
*/

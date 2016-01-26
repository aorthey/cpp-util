#include "util.h"

//test for util.h

using namespace util;
using namespace util::graph;

int main(void){
	Vector<float> b(3);

	Vector<int> a = VEC<int>( 3,1,4,5 );
	Matrix<int> m(4,4);
	m(0,0)=1;
	m(0,1)=2;
	m(0,2)=3;
	m(0,3)=4;
	m(1,0)=5;
	m(1,1)=6;
	m(1,2)=7;
	m(1,3)=8;
	
	util::cout << m << std::endl << a << std::endl << a << a.sum();
	util::cout << a << std::endl;


	a.push_back(3);
	
	util::cout << a << std::endl;

	//util::graph::Graph g("graph algorithm") = GRAPH(V("r"), V("b"));
	util::graph::Graph g = GRAPH(V("r"), V("b"));
	cout << g;
	Vector<V> vv = VEC<V>( V("a"), V("b"), V("c"), V("d"), V("e") );
	g.v = vv;
	Vector<E> ee = VEC<Edge>( 	E( g.v[0], g.v[1], 2 ), 
										E( g.v[0], g.v[0], 2 ), 
										E( g.v[1], g.v[2], 1 ), 
										E( g.v[1], g.v[3], 4 ), 
										E( g.v[2], g.v[3], 1 ), 
										E( g.v[2], g.v[0], 1 ), 
										E( g.v[3], g.v[2], 1 ), 
										E( g.v[3], g.v[4], 1 ) );
	g.e = ee;
	cout << g;
	g.backwardIteration( V("a") );
	//g.v = VEC<V>( V("a"), V("b"), V("c"), V("d"), V("e") );
	/* 
	util::graph::Graph g("graph algorithm");
	Vector<V> vv = VEC<V>( V("a"), V("b"), V("c"), V("d"), V("e") );
	util::cout << vv;
	g.setV(vv);
	//g.v = VEC<V>( V("a"), V("b"), V("c"), V("d"), V("e") );
	util::cout << g.v;
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
										*/

	
	return 0;
}

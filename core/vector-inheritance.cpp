#include <vector>
#include <cassert>
#include <algorithm>
#include <iostream>

namespace util{
static uint print_msg_counter = 0;
#define CUR_LOCATION "@" << __FILE__ << ":" << __LINE__
#define PRINT(msg) std::cout << "[" << print_msg_counter++ << "] " << \
        msg << " (" << CUR_LOCATION << ")" << std::endl
#define ABORT(msg) PRINT(msg); throw msg;
#define EXIT(msg) PRINT(msg); exit;
#define COUT(msg) PRINT(msg);

template<class T>
class Matrix: public std::vector<T>{
  public:
    explicit Matrix(){};
    explicit Matrix(int i, int j): std::vector<T>(i*j){d1=i;d2=j;dim=2;};


    T &operator()(int i, int j){
      return this->at(i*this->d2 + j);
    };

	  Matrix<T>& operator=(T rhs){
        typename std::vector<T>::const_iterator it;
        for(it=this->begin();it!=this->end();it++){
          this->at(it) = rhs;
        }
    }
	  Matrix<T>& operator=(Matrix<T> &rhs){

      if(rhs!=*this){
        this->d1 = rhs.d1;
        this->d2 = rhs.d2;

        typename std::vector<T>::const_iterator it;
        for(it=rhs.begin();it!=rhs.end();it++){
          this->at(it) = it;
        }

        return *rhs;
      }else{
        return *this;
      }

    }
	friend std::ostream &operator<<(std::ostream &os, Matrix<T> &m){

	  typename std::vector<T>::const_iterator it;
    uint c = 0;
    for(it=m.begin();it!=m.end();it++){
			//matrix [i*j]
			//uint j=c%m.d2;
			//uint i=c/m.d2;

			os << *it;
			if(c%m.d2 == m.d2-1 && c < m.d2*m.d1-1){
			  os << std::endl;
      }else{
        os << " ";
      }
      c++;
		}
		return os;
	}; 
	void eye(int i){
    assert(i==this->d1 && i==this->d2);
    for(uint k=0;k<i;k++){
      for(uint l=0;l<i;l++){
        uint c = k*this->d2 + l;
        if(k==l){
          this->at(c)=1;
        }else{
          this->at(c)=0;
        }
      }
    }
  }
	T sum(void){
		T sum=0;
	  typename std::vector<T>::const_iterator it;
		for(it=this->begin();it!=this->end();it++){
			sum+=*it;
		}
		return sum;
	}

	T prod(void){
		T prod=1;
	  typename std::vector<T>::const_iterator it;
		for(it=this->begin();it!=this->end();it++){
			prod*=*it;
		}
		return prod;
	}
  private:
    uint dim;
    int d1;
    int d2;
    //pMatrix *p;//pImpl
};

template<class __vector_template>
class Vector: public Matrix<__vector_template>{
public:	

  explicit Vector(int i): Matrix<__vector_template>(1,i){};


};
}

int main(){


	Vector<float> b(3);

	Vector<int> a(4);
	Matrix<int> m(4,4);
	//*/
	m(0,0)=1;
	m(0,1)=2;
	m(0,2)=3;
	m(0,3)=4;
	m(1,0)=5;
	m(1,1)=6;
	m(1,2)=7;
	m(1,3)=8;
	
	//m.eye(4);
  //*/
  COUT(m);
  COUT(a);
  COUT(b);

	a.push_back(3);
	
	COUT(a.sum());
	return 0;
}



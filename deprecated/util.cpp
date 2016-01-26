#include "util.h"
using namespace util;

template<class T>
Matrix<T>::Matrix(){this->d1=0;this->d2=0;this->dim=0;};

template<class T>
T& Matrix<T>::operator()(int i, int j){
	return this->at(i*this->d2 + j);
};

template<class T>
Matrix<T>& Matrix<T>::operator=(T rhs){
		uint d = this->size();
		this->clear();

		for(uint i=0;i<d;i++){
			this->push_back(rhs);
		}
		return *this;
}
template<class T>
Matrix<T>& Matrix<T>::operator=(Matrix<T> &rhs){

	if(&rhs!=this){
		this->d1 = rhs.d1;
		this->d2 = rhs.d2;
		this->dim = rhs.dim;

		typename std::vector<T>::const_iterator it;
		this->clear();

		for(it=rhs.begin();it!=rhs.end();it++){
			this->push_back(*it);
		}
		return *this;
	}else{
		return *this;
	}

}

template<class T>
void Matrix<T>::eye(int i){
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
template<class T>
T Matrix<T>::sum(void){
	T sum=0;
	typename std::vector<T>::const_iterator it;
	for(it=this->begin();it!=this->end();it++){
		sum+=*it;
	}
	return sum;
}

template<class T>
T Matrix<T>::prod(void){
	T prod=1;
	typename std::vector<T>::const_iterator it;
	for(it=this->begin();it!=this->end();it++){
		prod*=*it;
	}
	return prod;
};

#ifndef _SP_PAR_VEC_H_
#define _SP_PAR_VEC_H_

#include <iostream>
#include <vector>
#include <utility>

#ifdef NOTR1
	#include <boost/tr1/memory.hpp>
#else
	#include <tr1/memory>
#endif
#include "CommGrid.h"

using namespace std;
using namespace std::tr1;

template <class IT>
class DistEdgeList;


/** 
  * A sparse vector of length n (with nnz <= n of them being nonzeros) is distributed to 
  * diagonal processors in a way that respects ordering of the nonzero indices
  * Example: x = [5,1,6,2,9] for nnz(x)=5 and length(x)=10 
  *	we use 4 processors P_00, P_01, P_10, P_11
  * 	Then P_00 owns [1,2] (in the range [0...4]) and P_11 owns rest
  * In the case of A(v,w) type sparse matrix indexing, this doesn't matter because n = nnz
  * 	After all, A(v,w) will have dimensions length(v) x length (w) 
  * 	v and w will be of numerical type (NT) "int" and their indices (IT) will be consecutive integers 
  * It is possibly that nonzero counts are distributed unevenly
  * Example: x=[1,2,3,4,5] and length(x) = 10
  * Just like in SpParMat case, indices are local to processors (they belong to range [0,...,length-1] on each processor)
  *
  * TODO: Instead of repeated calls to "DiagWorld", this class should be oblivious to the communicator
  * 	  It should just distribute the vector to the MPI::IntraComm that it owns, whether diagonal or whole
 **/
  
template <class IT, class NT>
class SpParVec
{
public:
	SpParVec ( );
	SpParVec ( shared_ptr<CommGrid> grid);

	SpParVec<IT,NT> & operator+=(const SpParVec<IT,NT> & rhs);
	ifstream& ReadDistribute (ifstream& infile, int master);	

	void PrintInfo() const;
	void iota(IT size, NT first);
	SpParVec<IT,NT> operator() (const SpParVec<IT,IT> & ri) const;	// SpRef
	void SetElement (IT indx, NT numx);	// element-wise assignment

	// sort the vector itself
	// return the permutation vector
	SpParVec<IT, IT> sort();	

	IT getlocnnz() const 
	{
		return ind.size();
	}
	
	IT getnnz() const
	{
		IT totnnz = 0;
		IT locnnz = ind.size();
		(commGrid->GetDiagWorld()).Allreduce( &locnnz, & totnnz, 1, MPIType<IT>(), MPI::SUM); 
		return totnnz;
	}

	template <typename _UnaryOperation>
	void Apply(_UnaryOperation __unary_op)
	{
		transform(num.begin(), num.end(), num.begin(), __unary_op);
	}

private:
	shared_ptr<CommGrid> commGrid;
	vector< IT > ind;	// ind.size() give the number of nonzeros
	vector< NT > num;
	IT length;		// actual local length of the vector (including zeros)
	bool diagonal;
	const static IT zero = static_cast<IT>(0);

	template <class IU, class NU>
	friend class DenseParVec;
	
	template <class IU, class NU, class UDER>
	friend class SpParMat;

	template <typename IU>
	friend void RandPerm(SpParVec<IU,IU> & V, IU loclength); 	// called on an existing object, generates a random permutation
	
	template <typename IU>
	friend void RenameVertices(DistEdgeList<IU> & DEL);
};

#include "SpParVec.cpp"
#endif


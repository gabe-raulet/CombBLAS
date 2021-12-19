#include <mpi.h>
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <sstream>
#include <numeric>
#include "CombBLAS/CombBLAS.h"

template <class NT>
void PrintLocVec(std::vector<NT> vec)
{
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    std::cout << myrank+1 << ": ";
    std::copy(vec.begin(), vec.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;
}

int main(int argc, char *argv[])
{
    int nprocs, myrank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    {
        std::shared_ptr<combblas::CommGrid> fullWorld;
        fullWorld.reset(new combblas::CommGrid(MPI_COMM_WORLD, 0, 0));

        combblas::FullyDistVec<int, double> v(fullWorld);

        v.ParallelRead(std::string(argv[1]), true, combblas::maximum<double>());

        /* v.DebugPrint(); */

        auto commGrid = v.getcommgrid();
        MPI_Comm World = commGrid->GetWorld();
        MPI_Comm RowWorld = commGrid->GetRowWorld();
        MPI_Comm ColWorld = commGrid->GetColWorld();

        int rowneighs, rowrank;
        MPI_Comm_size(RowWorld, &rowneighs);
        MPI_Comm_rank(RowWorld, &rowrank);

        int mylocsize = v.LocArrSize();

        std::vector<int> rowvecs_counts(rowneighs, 0);
        std::vector<int> rowvecs_displs(rowneighs, 0);

        rowvecs_counts[rowrank] = mylocsize;

        MPI_Allgather(MPI_IN_PLACE, 0, MPI_INT, rowvecs_counts.data(), 1, MPI_INT, RowWorld);

        int rowvecs_size = std::accumulate(rowvecs_counts.begin(), rowvecs_counts.end(), static_cast<int>(0));

        std::partial_sum(rowvecs_counts.begin(), rowvecs_counts.end()-1, rowvecs_displs.begin()+1);

        std::vector<double> rowvecs(rowvecs_size);

        MPI_Allgatherv(v.GetLocArr(), mylocsize, MPI_DOUBLE, rowvecs.data(), rowvecs_counts.data(), rowvecs_displs.data(), MPI_DOUBLE, RowWorld);

        int complement_rank = commGrid->GetComplementRank();

        int complement_rowvecs_size = 0;
        MPI_Sendrecv(&rowvecs_size, 1, MPI_INT, complement_rank, TRX, &complement_rowvecs_size, 1, MPI_INT, complement_rank, TRX, World, MPI_STATUS_IGNORE);

        std::vector<double> complement_rowvecs(complement_rowvecs_size);
        MPI_Sendrecv(rowvecs.data(), rowvecs_size, MPI_DOUBLE, complement_rank, TRX, complement_rowvecs.data(), complement_rowvecs_size, MPI_DOUBLE, complement_rank, TRX, World, MPI_STATUS_IGNORE);

        PrintLocVec(rowvecs);
        PrintLocVec(complement_rowvecs);

    }

    MPI_Finalize();
    return 0;
}

// ===================
//  Author: lunasea & claude
//  date: 2026.06.24
// ===================

#pragma once

#include <mpi.h>
#include <cstddef>

namespace RI
{

// Manages an MPI_Win_allocate_shared window for intra-node zero-copy Tensor sharing.
// Each rank contributes n_local elements; all ranks can read any rank's segment via
// query_ptr(rank) after sync(). Non-owning ranks must treat the data as read-only.
//
// Lifetime contract: free() is a collective operation — call it at a collective
// synchronization point (e.g. LRI::free_tensors_map2). The destructor calls free()
// automatically; ensure all processes destroy their pool objects collectively.
template<typename T>
class SharedMemoryPool
{
public:
	SharedMemoryPool() = default;
	SharedMemoryPool(const SharedMemoryPool &) = delete;
	SharedMemoryPool & operator=(const SharedMemoryPool &) = delete;
	SharedMemoryPool(SharedMemoryPool && rhs);
	SharedMemoryPool & operator=(SharedMemoryPool && rhs);
	~SharedMemoryPool();

	// Collective: allocate n_local elements in the shared window.
	// node_comm must be an intra-node communicator (MPI_COMM_TYPE_SHARED).
	// Ranks with no owned data should pass n_local = 0.
	void allocate(const MPI_Comm & node_comm, std::size_t n_local);

	// Pointer to this rank's own segment; write data here before calling sync().
	T * local_ptr() const;

	// Local virtual address for rank r's window segment (valid after sync()).
	// Only meaningful when rank r allocated n_local > 0.
	T * query_ptr(int rank) const;

	// MPI_Win_fence(0): call after all writes, before non-owning ranks read.
	void sync();

	// Collective MPI_Win_free. Must be called at a collective synchronization point.
	void free();

	bool is_allocated() const;
	int node_rank() const;
	int node_size() const;

private:
	MPI_Win win_ = MPI_WIN_NULL;	// window, 
	T * local_ptr_ = nullptr;
	MPI_Comm node_comm_ = MPI_COMM_NULL;
	bool flag_allocated_ = false;
	int node_rank_ = -1;
	int node_size_ = -1;
};

}

#include "SharedMemoryPool.hpp"
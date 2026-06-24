// ===================
//  Author: lunasea & claude
//  date: 2026.06.24
// ===================

#pragma once

#include "SharedMemoryPool.h"
#include <stdexcept>
#include <string>

#define MPI_CHECK(x) if((x)!=MPI_SUCCESS)	throw std::runtime_error(std::string(__FILE__)+" line "+std::to_string(__LINE__));

namespace RI
{

template<typename T>
SharedMemoryPool<T>::SharedMemoryPool(SharedMemoryPool && rhs)
{
	this->free();
	this->win_ = rhs.win_;
	this->local_ptr_ = rhs.local_ptr_;
	this->node_comm_ = rhs.node_comm_;
	this->flag_allocated_ = rhs.flag_allocated_;
	this->node_rank_ = rhs.node_rank_;
	this->node_size_ = rhs.node_size_;
	rhs.flag_allocated_ = false;
}

template<typename T>
SharedMemoryPool<T> & SharedMemoryPool<T>::operator=(SharedMemoryPool && rhs)
{
	if(this != &rhs)
	{
		this->free();
		this->win_ = rhs.win_;
		this->local_ptr_ = rhs.local_ptr_;
		this->node_comm_ = rhs.node_comm_;
		this->flag_allocated_ = rhs.flag_allocated_;
		this->node_rank_ = rhs.node_rank_;
		this->node_size_ = rhs.node_size_;
		rhs.flag_allocated_ = false;
	}
	return *this;
}

template<typename T>
SharedMemoryPool<T>::~SharedMemoryPool()
{
	this->free();
}

template<typename T>
void SharedMemoryPool<T>::allocate(const MPI_Comm & node_comm, std::size_t n_local)
{
	if(this->flag_allocated_)
		return;
	this->node_comm_ = node_comm;
	MPI_CHECK( MPI_Comm_rank(node_comm, &this->node_rank_) );
	MPI_CHECK( MPI_Comm_size(node_comm, &this->node_size_) );
	void * raw_local = nullptr;
	MPI_CHECK( MPI_Win_allocate_shared(
		static_cast<MPI_Aint>(n_local) * static_cast<MPI_Aint>(sizeof(T)),
		static_cast<int>(sizeof(T)),	// bytes per unit
		MPI_INFO_NULL,
		this->node_comm_,
		&raw_local,	// [out] virtual logical addr. of the current proc. in its node, pointing to the same physical addr. for each node
		&this->win_) );
	this->local_ptr_ = static_cast<T *>(raw_local);
	this->flag_allocated_ = true;
}

template<typename T>
T * SharedMemoryPool<T>::local_ptr() const
{
	return this->local_ptr_;
}

template<typename T>
T * SharedMemoryPool<T>::query_ptr(int rank) const
{
	MPI_Aint size;
	int disp_unit;
	void * raw = nullptr;
	MPI_CHECK( MPI_Win_shared_query(this->win_, rank, &size, &disp_unit, &raw) );
	return static_cast<T *>(raw);
}

template<typename T>
void SharedMemoryPool<T>::sync()
{
	MPI_CHECK( MPI_Win_fence(0, this->win_) );
}

template<typename T>
void SharedMemoryPool<T>::free()
{
	if(this->flag_allocated_)
	{
		MPI_CHECK( MPI_Win_free(&this->win_) );
		this->flag_allocated_ = false;
		this->local_ptr_ = nullptr;
	}
}

template<typename T>
bool SharedMemoryPool<T>::is_allocated() const
{
	return this->flag_allocated_;
}

template<typename T>
int SharedMemoryPool<T>::node_rank() const
{
	return this->node_rank_;
}

template<typename T>
int SharedMemoryPool<T>::node_size() const
{
	return this->node_size_;
}

}

#undef MPI_CHECK
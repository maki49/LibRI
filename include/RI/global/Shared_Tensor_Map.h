#pragma once

#include "Tensor.h"
#include "Cereal_Types.h"
#include <cereal/archives/binary.hpp>
#include <mpi.h>
#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstring>
#include <functional>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#ifdef __linux__
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace RI
{
namespace Node_Shared
{
inline void check(int status)
{
    if(status != MPI_SUCCESS) throw std::runtime_error("LibRI node-shared MPI operation failed");
}

// A temporary collective scope, never retained by a Tensor or an LRI copy.
class Communicator
{
public:
    MPI_Comm comm = MPI_COMM_NULL;
    int rank = 0, size = 1;
    explicit Communicator(MPI_Comm parent)
    {
        check(MPI_Comm_split_type(parent, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &comm));
        check(MPI_Comm_rank(comm, &rank));
        check(MPI_Comm_size(comm, &size));
    }
    Communicator(const Communicator&) = delete;
    Communicator& operator=(const Communicator&) = delete;
    ~Communicator() { if(comm != MPI_COMM_NULL) MPI_Comm_free(&comm); }
};

template<typename Value>
void broadcast(const Communicator &node, Value &value, int root = 0)
{
    std::string bytes;
    if(node.rank == root)
    {
        std::ostringstream stream(std::ios::binary);
        { cereal::BinaryOutputArchive archive(stream); archive(value); }
        bytes = stream.str();
    }
    unsigned long long length = bytes.size();
    check(MPI_Bcast(&length, 1, MPI_UNSIGNED_LONG_LONG, root, node.comm));
    bytes.resize(static_cast<std::size_t>(length));
    for(std::size_t offset = 0; offset < bytes.size();)
    {
        const int count = static_cast<int>(std::min(bytes.size() - offset, std::size_t(INT_MAX)));
        check(MPI_Bcast(&bytes[offset], count, MPI_BYTE, root, node.comm));
        offset += count;
    }
    if(node.rank != root)
    {
        std::istringstream stream(bytes, std::ios::binary);
        cereal::BinaryInputArchive archive(stream); archive(value);
    }
}

// Only small key/shape or communication-judge metadata is exchanged here.
template<typename Value, typename Merge>
Value collect_on_root(const Communicator &node, const Value &local, Merge merge)
{
    Value result;
    for(int rank = 0; rank < node.size; ++rank)
    {
        Value next = node.rank == rank ? local : Value{};
        broadcast(node, next, rank);
        if(node.rank == 0) merge(result, next);
    }
    return result;
}

// A tmpfs-backed mapping, unlinked after all local ranks have opened it.
// MAP_PRIVATE shares physical pages on reads and gives process-local COW on
// writes. Destruction is just munmap: different Tensor owner counts are safe,
// including views that outlive MPI_Finalize. No MPI window is freed by a view.
class Mapping
{
public:
    void *address = nullptr;
    std::size_t bytes = 0;
    Mapping() = default;
    Mapping(const Mapping&) = delete;
    Mapping& operator=(const Mapping&) = delete;
    ~Mapping()
    {
#ifdef __linux__
        if(address) munmap(address, bytes);
#endif
    }

    template<typename Write>
    static std::shared_ptr<Mapping> create(const Communicator &node, std::size_t bytes, Write write)
    {
#ifndef __linux__
        throw std::runtime_error("LibRI node-shared inputs require Linux /dev/shm");
#else
        auto result = std::make_shared<Mapping>();
        if(!bytes) return result;
        int fd = -1, error = 0;
        std::string filename;
        if(node.rank == 0)
        {
            char pattern[] = "/dev/shm/libri-input-XXXXXX";
            fd = mkstemp(pattern);
            if(fd < 0) error = errno;
            else
            {
                filename = pattern;
                // Reserve tmpfs space before touching pages, so lack of space
                // is a collective error rather than a SIGBUS on the writer.
                if(bytes > static_cast<std::size_t>(std::numeric_limits<off_t>::max())) error = EFBIG;
                else error = posix_fallocate(fd, 0, static_cast<off_t>(bytes));
                if(!error)
                {
                    void *writer = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
                    if(writer == MAP_FAILED) error = errno;
                    else
                    {
                        try { write(writer); }
                        catch(...) { error = EIO; }
                        munmap(writer, bytes);
                    }
                }
            }
        }
        check(MPI_Bcast(&error, 1, MPI_INT, 0, node.comm));
        if(error)
        {
            if(fd >= 0) { close(fd); unlink(filename.c_str()); }
            throw std::runtime_error("LibRI shared input allocation failed: " + std::string(std::strerror(error)));
        }
        broadcast(node, filename);
        if(node.rank != 0) fd = open(filename.c_str(), O_RDONLY);
        if(fd < 0) error = errno;
        else
        {
            result->address = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
            if(result->address == MAP_FAILED) { result->address = nullptr; error = errno; }
            else result->bytes = bytes;
            close(fd);
        }
        int any_error = 0;
        check(MPI_Allreduce(&error, &any_error, 1, MPI_INT, MPI_MAX, node.comm));
        // All opens/mappings have completed before unlinking the file name.
        if(node.rank == 0) unlink(filename.c_str());
        if(any_error) throw std::runtime_error("LibRI shared input mapping failed: " + std::string(std::strerror(any_error)));
        return result;
#endif
    }
};

// root_values contains the union of this node's requested blocks on rank 0.
// Keep each rank's original key selection; only numerical storage is shared.
template<typename TA, typename TAC, typename T, typename Judge>
std::map<TA, std::map<TAC, Tensor<T>>> share_map(
    const Communicator &node, std::map<TA, std::map<TAC, Tensor<T>>> root_values,
    const Judge &local_judge)
{
    static_assert(std::is_trivially_copyable<T>::value, "Shared tensor values must be trivially copyable");
    using Layout = std::map<TA, std::map<TAC, std::pair<std::size_t, Shape_Vector>>>;
    Layout layout;
    std::size_t count = 0;
    int overflow = 0;
    if(node.rank == 0)
        for(const auto &row : root_values)
            for(const auto &entry : row.second)
            {
                const std::size_t n = entry.second.shape.get_shape_all();
                if(n > std::numeric_limits<std::size_t>::max()/sizeof(T) - count) overflow = 1;
                if(!overflow) { layout[row.first][entry.first] = {count, entry.second.shape}; count += n; }
            }
    check(MPI_Bcast(&overflow, 1, MPI_INT, 0, node.comm));
    if(overflow) throw std::overflow_error("LibRI shared tensor size overflow");
    broadcast(node, count);
    broadcast(node, layout);
    auto mapping = Mapping::create(node, count * sizeof(T), [&](void *memory)
    {
        T *destination = static_cast<T*>(memory);
        for(auto &row : root_values)
            for(auto &entry : row.second)
            {
                const auto &block = layout.at(row.first).at(entry.first);
                const std::size_t n = block.second.get_shape_all();
                if(n) std::memcpy(destination + block.first, entry.second.ptr(), n * sizeof(T));
                // Drop the root's private receive buffer as each block is copied.
                entry.second = Tensor<T>{};
            }
        root_values.clear();
    });
    std::map<TA, std::map<TAC, Tensor<T>>> result;
    for(const auto &row : layout)
        for(const auto &entry : row.second)
            if(local_judge.judge(std::make_tuple(row.first, entry.first)))
            {
                T *base = static_cast<T*>(mapping->address);
                result[row.first][entry.first] = Tensor<T>(entry.second.second,
                    base ? base + entry.second.first : nullptr, mapping);
            }
    return result;
}
}
}

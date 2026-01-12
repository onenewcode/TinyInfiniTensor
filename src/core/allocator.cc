#include "core/allocator.h"
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = this->getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================
        
        for (auto it = free_blocks.begin(); it != free_blocks.end(); ++it) {
            if (it->second >= size) {
                size_t addr = it->first;
                if (it->second == size) {
                    free_blocks.erase(it);
                } else {
                    free_blocks[it->first + size] = it->second - size;
                    free_blocks.erase(it);
                }
                used += size;
                return addr;
            }
        }
        
        size_t addr = used;
        used += size;
        if (used > peak) {
            peak = used;
        }
        
        return addr;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================
        
        size_t end = addr + size;
        
        auto next_it = free_blocks.lower_bound(addr);
        
        auto prev_it = (next_it != free_blocks.begin()) ? std::prev(next_it) : free_blocks.end();
        
        bool merge_prev = (prev_it != free_blocks.end()) && (prev_it->first + prev_it->second == addr);
        bool merge_next = (next_it != free_blocks.end()) && (end == next_it->first);
        
        if (merge_prev && merge_next) {
            prev_it->second = prev_it->second + size + next_it->second;
            free_blocks.erase(next_it);
        } else if (merge_prev) {
            prev_it->second += size;
        } else if (merge_next) {
            size_t new_size = size + next_it->second;
            free_blocks.erase(next_it);
            free_blocks[addr] = new_size;
        } else {
            free_blocks[addr] = size;
        }
        
        used -= size;
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}

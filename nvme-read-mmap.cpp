#include <sys/mman.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <string>
#include <iostream>
#include <filesystem>
#include <memory>
#include <cstring>
#include <chrono>
#include <vector>
#include <thread>
#include <cstdint>
#include <sstream>
namespace fs = std::filesystem;

const std::string gc_src_dir = "/scratch/LLMs/models/bloom";
constexpr size_t gc_page_size = 4096;
constexpr uintptr_t gc_page_mask = gc_page_size - 1;
constexpr size_t gc_n_workers = 4096;

struct Task {
    const uint8_t* partition_;
    const size_t n_pages_;
    Task(const uint8_t* partition, const size_t n_pages)
        : partition_(partition), n_pages_(n_pages) { }
};

bool EndsWith(const std::string& text, const std::string& suffix) {
    if(text.size() < suffix.size()) {
        return false;
    }
    return memcmp(text.data()+text.size()-suffix.size(), suffix.data(), suffix.size()) == 0;
}

void WorkerEntry(const uint8_t* partition, const size_t n_pages) {
    uintptr_t lim_addr = uintptr_t(partition) + n_pages * gc_page_size;
    lim_addr &= ~((uintptr_t(1)<<12) - 1);
    const size_t length = lim_addr - uintptr_t(partition);
    if(mlock(partition, length) == -1) {
        std::stringstream ss;
        ss << "Failed to pin a partition at " << static_cast<const void*>(partition) << " length=" << length << " ";
        const std::string message = ss.str();
        std::cout << message;
        perror("Faild to pin a partition");
    }
}

void MultiLock(const uint8_t* buffer, const size_t n_bytes) {
    //std::vector<std::thread> workers;
    const size_t n_pages = (n_bytes + gc_page_size - 1) / gc_page_size;
    const size_t pages_per_worker = (n_pages + gc_n_workers - 1) / gc_n_workers;
#pragma omp parallel for num_threads(gc_n_workers)
    for(size_t i=0; i<gc_n_workers; i++) {
        const size_t first_page = i * pages_per_worker;
        const size_t limit_page = std::min(n_pages, (i+1)*pages_per_worker);
        if(first_page >= limit_page) {
            continue;
        }
        //workers.emplace_back(&WorkerEntry, buffer + first_page * gc_page_size, limit_page - first_page);
        WorkerEntry(buffer + first_page * gc_page_size, limit_page - first_page);
    }
    // for(auto& worker : workers) {
    //     worker.join();
    // }
}

const uint8_t* AlignPageUp(const void* ptr) {
    return reinterpret_cast<uint8_t*>((uintptr_t(ptr) + gc_page_mask) & (~gc_page_mask));
}

uint8_t* AlignPageUp(void* ptr) {
    return const_cast<uint8_t*>(AlignPageUp(static_cast<const void*>(ptr)));
}

const uint8_t* AlignPageDown(const void* ptr) {
    return reinterpret_cast<uint8_t*>(uintptr_t(ptr) & (~gc_page_mask));
}

void MultiRead(const uint8_t* buffer, const size_t n_bytes) {
    void* raw_storage = malloc((gc_n_workers+1) * gc_page_size);
    uint8_t* storage = AlignPageUp(raw_storage);
    const uint8_t* begin_buf = AlignPageUp(buffer);
    const uint8_t* end_buf = AlignPageDown(buffer + n_bytes);
    const size_t n_pages = (end_buf - begin_buf) / gc_page_size;
    const size_t pages_per_worker = (n_pages + gc_n_workers - 1) / gc_n_workers;
#pragma omp parallel for num_threads(gc_n_workers)
    for(size_t i=0; i<gc_n_workers; i++) {
        if(i == 0 && buffer < begin_buf) {
            memcpy(storage + i*gc_page_size, buffer, begin_buf - buffer);
        }
        const size_t first_page = i * pages_per_worker;
        const size_t limit_page = std::min(n_pages, (i+1)*pages_per_worker);
        for(size_t j=first_page; j<limit_page; j++) {
            memcpy(storage + i*gc_page_size, begin_buf + j*gc_page_size, gc_page_size);
        }
        if(i == gc_n_workers-1 && end_buf < buffer + n_bytes) {
            memcpy(storage + i*gc_page_size, end_buf, buffer + n_bytes - end_buf);
        }
    }
    free(raw_storage);
}

void MultiLargeRead(const uint8_t* buffer, const size_t n_bytes) {
    uint8_t* storage = static_cast<uint8_t*>(malloc(n_bytes));
    const size_t bytes_per_worker = (n_bytes + gc_n_workers - 1) / gc_n_workers;
#pragma omp parallel for num_threads(gc_n_workers)
    for(size_t i=0; i<gc_n_workers; i++) {
        const size_t first_byte = i * bytes_per_worker;
        const size_t limit_byte = std::min(n_bytes, (i+1)*bytes_per_worker);
        if(first_byte >= limit_byte) {
            continue;
        }
        memcpy(storage + first_byte, buffer + first_byte, limit_byte - first_byte);
    }
    free(storage);
}

int main() {
    for (const auto & entry : fs::directory_iterator(gc_src_dir)) {
        if(!EndsWith(entry.path(), ".safetensors") && !EndsWith(entry.path(), ".bin")) {
            continue;
        }
        std::cout << entry.path() << std::endl;

        const int fd = open(entry.path().c_str(), O_RDONLY | O_LARGEFILE | O_DIRECT);
        if(fd == -1) {
            perror("Failed to open");
            continue;
        }
        struct stat file_stat;
        if (fstat(fd, &file_stat) == -1) {
            perror("Faild to stat");
            continue;
        }
        //posix_fadvise(fd, 0, file_stat.st_size, POSIX_FADV_RANDOM);
        void* buffer = mmap(nullptr, file_stat.st_size, PROT_READ, MAP_PRIVATE | MAP_NORESERVE, fd, 0);
        if(buffer == MAP_FAILED) {
            perror("Failed to map");
            close(fd);
            continue;
        }
        // if(madvise(buffer, file_stat.st_size, MADV_SEQUENTIAL) == -1) {
        //     perror("Failed madvise(Sequential)");
        // }
        // if(mlock(buffer, file_stat.st_size) == -1) {
        //     perror("Failed to pin memory");
        //     munmap(buffer, file_stat.st_size);
        //     close(fd);
        //     continue;
        // }
        // MultiLock(static_cast<uint8_t*>(buffer), file_stat.st_size);
        // if(munlock(buffer, file_stat.st_size) == -1) {
        //     perror("Failed to unpin memory");
        // }

        std::chrono::steady_clock::time_point tmLast = std::chrono::steady_clock::now();
        MultiLargeRead(static_cast<uint8_t*>(buffer), file_stat.st_size);
        std::chrono::steady_clock::time_point tmNow = std::chrono::steady_clock::now();
        const double nSec = std::chrono::duration_cast<std::chrono::nanoseconds>(tmNow - tmLast).count() / 1e9;
        const double GBperSec = (file_stat.st_size / nSec) / 1e9;
        std::cout << file_stat.st_size << " bytes in " << nSec << " seconds: "
            << GBperSec << " billion bytes per second." << std::endl;

        if(munmap(buffer, file_stat.st_size) == -1) {
            perror("Failed to unmap");
        }
        posix_fadvise(fd, 0, file_stat.st_size, POSIX_FADV_DONTNEED);
        if(close(fd) == -1) {
            perror("Failed to close");
        }
    }
    return 0;
}

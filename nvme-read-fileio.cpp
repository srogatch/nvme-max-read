#include <sys/mman.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/uio.h>

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
constexpr size_t gc_n_workers = 64;

bool EndsWith(const std::string& text, const std::string& suffix) {
    if(text.size() < suffix.size()) {
        return false;
    }
    return memcmp(text.data()+text.size()-suffix.size(), suffix.data(), suffix.size()) == 0;
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

void Scattered(const int fd, const size_t n_bytes) {
    // See https://stackoverflow.com/questions/27271801/c-the-permitted-maximum-of-the-iovcnt-argument-in-writev
    constexpr size_t buffers_per_worker = 1024;
    constexpr size_t n_buffers = gc_n_workers * buffers_per_worker;
    const size_t even_page_bytes = (n_bytes + gc_page_mask) & (~gc_page_mask);
    size_t bytes_per_buffer = (n_bytes + n_buffers - 1) / n_buffers;
    bytes_per_buffer = (bytes_per_buffer + gc_page_mask) & (~gc_page_mask);
    const size_t used_buffers = (n_bytes + bytes_per_buffer - 1) / bytes_per_buffer;
    void* raw_storage = malloc(bytes_per_buffer * used_buffers + gc_page_mask);
    uint8_t* storage = AlignPageUp(raw_storage);
    iovec buffers[n_buffers];

#pragma omp parallel for
    for(size_t i=0; i<used_buffers; i++) {
        const size_t first_byte = bytes_per_buffer * i;
        const size_t limit_byte = std::min(even_page_bytes, bytes_per_buffer * (i+1));
        buffers[i].iov_len = limit_byte - first_byte;
        buffers[i].iov_base = storage + first_byte;
    }

#pragma omp parallel for num_threads(gc_n_workers)
    for(size_t i_worker = 0; i_worker < gc_n_workers; i_worker++) {
        const size_t first_buffer = i_worker * buffers_per_worker;
        const size_t limit_buffer = std::min((i_worker+1) * buffers_per_worker, used_buffers);
        if(first_buffer >= limit_buffer) {
            continue;
        }
        ssize_t n_read = preadv2(fd, buffers + first_buffer, limit_buffer - first_buffer, 0, RWF_HIPRI);
        if(n_read == -1) {
            perror("Failed to read file");
        }
    }
    free(raw_storage);
}

void MultiLargeRead(const int fd, const size_t n_bytes) {
    void* raw_storage = malloc(n_bytes + gc_page_size + gc_page_mask);
    uint8_t *storage = AlignPageUp(raw_storage);
    size_t bytes_per_worker = (n_bytes + gc_n_workers - 1) / gc_n_workers;
    bytes_per_worker = (bytes_per_worker + gc_page_mask) & (~gc_page_mask);
#pragma omp parallel for num_threads(gc_n_workers)
    for(size_t i=0; i<gc_n_workers; i++) {
        const size_t first_byte = i * bytes_per_worker;
        const size_t limit_byte = std::min(n_bytes, (i+1)*bytes_per_worker);
        if(first_byte >= limit_byte) {
            continue;
        }
        const ssize_t n_read = pread(fd, storage + first_byte,
            (limit_byte - first_byte + gc_page_mask) & (~gc_page_mask),
            first_byte);
        if(n_read == -1) {
            perror("Faild to read file");
        }
    }
    free(raw_storage);
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

        std::chrono::steady_clock::time_point tmLast = std::chrono::steady_clock::now();
        Scattered(fd, file_stat.st_size);
        std::chrono::steady_clock::time_point tmNow = std::chrono::steady_clock::now();
        const double nSec = std::chrono::duration_cast<std::chrono::nanoseconds>(tmNow - tmLast).count() / 1e9;
        const double GBperSec = (file_stat.st_size / nSec) / 1e9;
        std::cout << file_stat.st_size << " bytes in " << nSec << " seconds: "
            << GBperSec << " billion bytes per second." << std::endl;

        posix_fadvise(fd, 0, file_stat.st_size, POSIX_FADV_DONTNEED);
        if(close(fd) == -1) {
            perror("Failed to close");
        }
    }
    return 0;
}

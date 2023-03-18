mkdir build
g++ -fopenmp -O3 nvme-read-mmap.cpp -o build/nvme-read-mmap
echo 3 | sudo tee /proc/sys/vm/drop_caches
./build/nvme-read-mmap

mkdir build
g++ -fopenmp -O3 nvme-read-fileio.cpp -o build/nvme-read-fileio
echo 3 | sudo tee /proc/sys/vm/drop_caches
./build/nvme-read-fileio

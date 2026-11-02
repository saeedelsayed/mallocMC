/*
  mallocMC: Memory Allocator for Many Core Architectures.
  https://www.hzdr.de/crp

  Copyright 2014 - 2024 Institute of Radiation Physics,
                 Helmholtz-Zentrum Dresden - Rossendorf

  Author(s):  Carlchristian Eckert - c.eckert ( at ) hzdr.de
              Julian Lenz - j.lenz ( at ) hzdr.de

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.
*/

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

#include <mallocMC/mallocMC.hpp>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <chrono>
#include <fstream>
#include <nlohmann/json.hpp>


using json = nlohmann::json;

using mallocMC::CreationPolicies::FlatterScatter;

using Dim = alpaka::DimInt<1>;
using Idx = std::size_t;

// Define the device accelerator
using Acc = alpaka::ExampleDefaultAcc<Dim, Idx>;

constexpr uint32_t const blocksize = 128U * 1024U * 1024U;
constexpr uint32_t const pagesize = 128U * 1024U;
constexpr uint32_t const wasteFactor = 1U;
uint32_t allocSize = 3200U;
uint32_t noOfMeasurements = 1U;
uint32_t maxNoOfTHreads = 1U;
uint32_t noOfChunks = 31;
uint32_t noOfPages = 4095;
uint32_t requiredPages = 1;
uint32_t fittedThreadsInBlock = noOfPages / requiredPages;
uint32_t maskSize = 32;


// This happens to also work for the original Scatter algorithm, so we only define one.
struct FlatterScatterHeapConfig : FlatterScatter<>::Properties::HeapConfig
{
    static constexpr auto accessblocksize = blocksize;
    static constexpr auto pagesize = ::pagesize;
    static constexpr auto heapsize = 2U * 1024U * 1024U * 1024U;
    // Only used by original Scatter (but it doesn't hurt FlatterScatter to keep):
    static constexpr auto regionsize = 16;
    static constexpr auto wastefactor = wasteFactor;
};

// struct FlatterScatterHashConfig : FlatterScatter<>::Properties::HashConfig
// {
//     static constexpr uint32_t blockStride = 0;

//     template<uint32_t T_pageSize, typename TAcc>
//     ALPAKA_FN_INLINE ALPAKA_FN_ACC static auto hash(TAcc const& acc, uint32_t const numBytes) -> uint32_t
//     {
//         return 0;
//     }
// };

struct XMallocConfig
{
    static constexpr auto pagesize = FlatterScatterHeapConfig::pagesize;
};

struct ShrinkConfig
{
    static constexpr auto dataAlignment = 16;
};


ALPAKA_STATIC_ACC_MEM_GLOBAL int** arA;


template<
    typename T_CreationPolicy,
    typename T_ReservePoolPolicy,
    typename T_AlignmentPolicy = mallocMC::AlignmentPolicies::Shrink<ShrinkConfig>>
auto maxCollisionHashFun() -> int
{
    using Allocator = mallocMC::Allocator<
        alpaka::AccToTag<Acc>,
        T_CreationPolicy,
        mallocMC::DistributionPolicies::Noop,
        mallocMC::OOMPolicies::ReturnNull,
        T_ReservePoolPolicy,
        T_AlignmentPolicy>;

    auto const platform = alpaka::Platform<Acc>{};
    auto const dev = alpaka::getDevByIdx(platform, 0);
    auto queue = alpaka::Queue<Acc, alpaka::Blocking>{dev};

    // init the heap
    std::cerr << "initHeap...";
    auto const heapSize = 2U * 1024U * 1024U * 1024U;
    std::cerr << "done\n";
    std::cout << Allocator::info("\n") << '\n';

    // create arrays of arrays on the device
    {
        int globalAtomicsSum = 0;

        auto allocArray
            = [] ALPAKA_FN_ACC(Acc const& acc, int x, uint16_t* warpArray, uint16_t* smArray, uint16_t* allocSizeArray, Allocator::AllocatorHandle allocHandle)
        {
            arA<Acc> = static_cast<int**>(allocHandle.malloc(acc, sizeof(int*) * x, warpArray, smArray, allocSizeArray));
        };

        auto freeArray
            = [] ALPAKA_FN_ACC(Acc const& acc, Allocator::AllocatorHandle allocHandle)
        {
            allocHandle.free(acc, arA<Acc>);
        };

        auto allocMemory
            = [] ALPAKA_FN_ACC(Acc const& acc, uint16_t* warpArray, uint16_t* smArray, uint16_t* allocSizeArray, Allocator::AllocatorHandle allocHandle)
        {
            auto const id = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
            // printf("alloc ");
            arA<Acc>[id] = static_cast<int*>(allocHandle.malloc(acc, 16, warpArray, smArray, allocSizeArray));
        };

        auto freeMemory
            = [] ALPAKA_FN_ACC(Acc const& acc, Allocator::AllocatorHandle allocHandle)
        {
            auto const id = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0]; 
            // printf("free ");  
            allocHandle.free(acc, arA<Acc>[id]);
        };

        Allocator scatterAlloc(dev, queue, heapSize); // 2GB for device-side malloc
        std::vector<uint32_t> arrayOfThreads(maxNoOfTHreads);

        // uint32_t noOfThreads = 1;
        // for(auto& thread : arrayOfThreads)
        // {
        //     thread = noOfThreads;
        //     noOfThreads = noOfThreads + 1;
        //     // printf("thread count: %d\n", thread);
        // }

        static constexpr uint32_t threadsCount = 1000000;


        auto warpArrayDevice = alpaka::allocBuf<uint16_t, Idx>(dev, Idx{threadsCount});
        auto smArrayDevice = alpaka::allocBuf<uint16_t, Idx>(dev, Idx{threadsCount});
        auto allocSizeArrayDevice = alpaka::allocBuf<uint16_t, Idx>(dev, Idx{threadsCount});

        auto const hostPlatform = alpaka::Platform<alpaka::DevCpu>{};
        auto const hostDev = alpaka::getDevByIdx(hostPlatform, 0);
        auto warpArrayHost = alpaka::allocBuf<uint16_t, Idx>(hostDev, Idx{threadsCount});
        auto smArrayHost = alpaka::allocBuf<uint16_t, Idx>(hostDev, Idx{threadsCount});
        auto allocSizeArrayHost = alpaka::allocBuf<uint16_t, Idx>(hostDev, Idx{threadsCount});

        auto* warpArray = alpaka::getPtrNative(warpArrayDevice);
        auto* smArray = alpaka::getPtrNative(smArrayDevice);
        auto* allocSizeArray = alpaka::getPtrNative(allocSizeArrayDevice);

        std::fstream gpuData;
        gpuData.open("threadIDs.csv", std::ios::out | std::ios::app);
        gpuData << "thread id, warp id, sm id, alloc size\n";

        std::ofstream csv("results.csv");
        csv << "thread_count";
        for(int i = 1; i <= noOfMeasurements; ++i)
            csv << ",measurement" << i;
        csv << ",mean";
        csv << ",noOfAtmoics\n";

        Allocator::CreationPolicy::template 
        initHeap<Acc>(dev, queue, scatterAlloc.getAllocatorHandle().devAllocator, scatterAlloc.getHeapLocations()[0].p ,heapSize);

        auto const workDivSingle = alpaka::WorkDivMembers<Dim, Idx>{Idx{1}, Idx{1}, Idx{1}};
        // printf("alloc size: %d\n", allocSize);

        // for(auto const& thread: arrayOfThreads)
        // {

            // uint32_t thread = 1;
            auto const workDiv = alpaka::WorkDivMembers<Dim, Idx>{Idx{threadsCount}, Idx{1}, Idx{1}};
            std::cout << "Number Of Threads: " << threadsCount << '\n';
            std::vector<std::chrono::duration<double>> measurements(noOfMeasurements);

            for(auto& measurement : measurements)
            {

            // auto bufDevice = allocBuf<int*, Idx>(dev, sizeof(int*) * threadsCount);
            // auto arrayPointer = alpaka::getPtrNative(bufDevice);
                alpaka::enqueue(
                    queue,
                    alpaka::createTaskKernel<Acc>(
                        workDivSingle,
                        allocArray,
                        threadsCount,
                        warpArray,
                        smArray,
                        allocSizeArray,
                        scatterAlloc.getAllocatorHandle()
                    )
                );
                
                const auto start = std::chrono::high_resolution_clock::now();

                alpaka::enqueue(
                    queue,
                    alpaka::createTaskKernel<Acc>(
                        workDiv,
                        allocMemory,
                        warpArray,
                        smArray,
                        allocSizeArray,
                        scatterAlloc.getAllocatorHandle()
                    )
                );

                const auto end = std::chrono::high_resolution_clock::now();
                measurement = end - start;

                alpaka::enqueue(
                    queue,
                    alpaka::createTaskKernel<Acc>(
                        workDiv,
                        freeMemory,
                        scatterAlloc.getAllocatorHandle()
                    )
                );
                
                // alpaka::enqueue(
                //     queue,
                //     alpaka::createTaskKernel<Acc>(
                //         workDivSingle,
                //         freeArray,
                //         scatterAlloc.getAllocatorHandle()
                //     )
                // );

            }
            alpaka::memcpy(queue, warpArrayHost, warpArrayDevice, Idx{threadsCount});
            alpaka::memcpy(queue, smArrayHost, smArrayDevice, Idx{threadsCount});
            alpaka::memcpy(queue, allocSizeArrayHost, allocSizeArrayDevice, Idx{threadsCount});
            alpaka::wait(queue);

            auto const* warpArrayData = alpaka::getPtrNative(warpArrayHost);
            auto const* smArrayData = alpaka::getPtrNative(smArrayHost);
            auto const* allocSizeArrayData = alpaka::getPtrNative(allocSizeArrayHost);

            // std::copy(measurements.cbegin(), measurements.cend(), std::ostream_iterator<std::chrono::duration<double>>(std::cout, ", "));
            // std::cout << "\n";
            auto mean = std::reduce(measurements.cbegin(), measurements.cend()) / noOfMeasurements;
            // std::cout << "mean: " << mean << "\n";

            int localAtomicsSum = 0;

            // for(int i = 0; i < thread; i++)
            // {
            //     int threadLocalIndex = i % fittedThreadsInBlock; 
            //     int insideBlock = threadLocalIndex * requiredPages + 2 * requiredPages;
            //     int outsizeBlock = (i / fittedThreadsInBlock) * (noOfPages - requiredPages + 1);
            //     int sum = insideBlock + outsizeBlock; 
            //     localAtomicsSum += sum;
            // }
            
            // localAtomicsSum *= noOfMeasurements;
            // globalAtomicsSum += localAtomicsSum;


            csv << threadsCount;
            for(auto const& m : measurements)
                csv << "," << m.count();
            csv << "," << mean.count();
            csv << "," << localAtomicsSum/noOfMeasurements << "\n";

            for(int i = 0; i < threadsCount; i++)
            {
                gpuData << i << ", ";
                gpuData << warpArrayData[i] << ", ";
                gpuData << smArrayData[i] << ", ";
                gpuData << allocSizeArrayData[i] << "\n";
            }

        // std::cout << "Total number of atmoic operations: " << globalAtomicsSum << '\n';

    }

    return 0;
}

auto main(int argc, char* argv[]) -> int
{
    if(argc != 2)
    {
        std::cerr << "Usage: benchmark config.json\n";
        return 1;
    }

    std::string filename;
    filename = argv[1];

    std::ifstream file(filename);

    json j;
    file >> j;

    allocSize = j["allocSize"];
    noOfMeasurements = j["noOfMeasurements"];
    maxNoOfTHreads = j["maxNoOfTHreads"];
    noOfChunks = j["noOfChunks"];
    noOfPages = j["noOfPages"];
    requiredPages = j["requiredPages"];
    maskSize = j["maskSize"];
    fittedThreadsInBlock = noOfPages / requiredPages;

    // allocSize = 32U;
    
    maxCollisionHashFun<FlatterScatter<FlatterScatterHeapConfig/*, FlatterScatterHashConfig*/>, mallocMC::ReservePoolPolicies::AlpakaBuf<Acc>>();

    return 0;
}

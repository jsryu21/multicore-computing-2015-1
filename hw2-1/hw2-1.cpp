#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <typeinfo>
#include <pthread.h>
#include <cfloat>
#include <climits>
#include <limits>
#include <cmath>
#include <map>

template< typename T >
class GenerateDatas
{
    public:
        GenerateDatas(const int size = 10000, const T low = 0, const T high = 1000000000000000000)
        {
            datas = std::vector< T >(size);
            std::srand(std::time(NULL));
            for (int i = 0; i < size; ++i)
            {
                // http://stackoverflow.com/questions/686353/c-random-float-number-generation
                datas[i] = low + static_cast< T >(std::rand()) / static_cast< T >(RAND_MAX / (high - low));
            }
            // accumulate for linear vector
            for (int i = 1; i < size; ++i)
            {
                datas[i] += datas[i - 1];
            }
        }
        void PrintDatas(int printCnt = 5) const
        {
            // Print generated datas in the increasing order
            std::cout << "Increasing order : ";
            for (int i = 0; i < std::min(printCnt, static_cast< int >(datas.size())); ++i)
            {
                std::cout << datas[i] << " ";
            }
            std::cout << "... ";
            for (int i = std::max(printCnt, static_cast< int >(datas.size() - printCnt)); i < datas.size(); ++i)
            {
                std::cout << datas[i] << " ";
            }
            std::cout << std::endl;
        }
        void PrintReversedDatas(int printCnt = 5) const
        {
            // Print generated datas in the decreasing order
            std::cout << "Decreasing order : ";
            for (int i = datas.size(); i > std::max(static_cast< int >(datas.size() - printCnt), 0); --i)
            {
                std::cout << datas[i - 1] << " ";
            }
            std::cout << "... ";
            for (int i = std::min(printCnt, static_cast< int >(datas.size() - printCnt)); i > 0; --i)
            {
                std::cout << datas[i - 1] << " ";
            }
            std::cout << std::endl;
        }
        template< typename A > A AccumulateDatas()
        {
            // Accumulate datas to datatype A variable
            A target = static_cast< A >(0);
            for (int i = 0; i < datas.size(); ++i)
            {
                target += static_cast< A >(datas[i]);
            }
            return target;
        }
        template< typename A > A AccumulateReversedDatas()
        {
            // Accumulate datas reversely to datatype A variable
            A target = static_cast< A >(0);
            for (int i = datas.size(); i > 0; --i)
            {
                target += static_cast< A >(datas[i - 1]);
            }
            return target;
        }
    private:
        std::vector< T > datas;
};

template< typename T >
struct ThreadData
{
    // used for passing arguments to pthread
    int Size;
    T Low;
    T High;
};

template< typename T, typename B, typename G >
struct Status
{
    // used for returning values from pthread
    int Size;
    T TInc;
    T TDec;
    B BInc;
    B BDec;
    G GInc;
    G GDec;
};

template< typename T >
class ThreadDataGenerator
{
    public:
        ThreadDataGenerator()
        {
            for (int i = 1; i < 11; ++i)
            {
                ThreadData< T > threadData;
                threadData.Size = std::pow(2, i);
                threadData.Low = 0;
                threadData.High = static_cast< T >(std::numeric_limits< T >::max()) / (static_cast< T >(threadData.Size) * static_cast< T >(threadData.Size) * 2);
                threadDatas.push_back(threadData);
            }
        }
        const std::vector< ThreadData< T > >& GetThreadDatas() const
        {
            return threadDatas;
        }
        void PrintThreadDatas() const
        {
            for (int i = 0; i < threadDatas.size(); ++i)
            {
                std::cout << threadDatas[i].size << " " << threadDatas[i].low << " " << threadDatas[i].high << std::endl;
            }
        }
    private:
        std::vector< ThreadData< T > > threadDatas;
};

void* RunGenerator(void* threadData)
{
    ThreadData< float >* tD = reinterpret_cast< ThreadData< float >* >(threadData);
    GenerateDatas< float > generator(tD->Size, tD->Low, tD->High);
    //generator.PrintDatas();
    //generator.PrintReversedDatas();
    Status< float, double, long double >* ret = new Status< float, double, long double >();
    ret->Size = tD->Size;
    ret->TInc = generator.AccumulateDatas< float >();
    ret->TDec = generator.AccumulateReversedDatas< float >();
    ret->BInc = generator.AccumulateDatas< double >();
    ret->BDec = generator.AccumulateReversedDatas< double >();
    ret->GInc = generator.AccumulateDatas< long double >();
    ret->GDec = generator.AccumulateReversedDatas< long double >();
    pthread_exit(ret);
}

int main()
{
    std::map< int, std::map< int, int > > result;
    for (int k = 0; k < 100000; ++k)
    {
        // http://www.tutorialspoint.com/cplusplus/cpp_multithreading.htm
        // http://anow.tistory.com/144
        ThreadDataGenerator< float > tDG;
        std::vector< ThreadData< float > > threadDatas = tDG.GetThreadDatas();
        //tDG.PrintThreadDatas();
        int numThreads = threadDatas.size();
        pthread_t threads[numThreads];
        pthread_attr_t attr;
        void *status;

        // Initialize and set thread joinable
        pthread_attr_init(&attr);
        pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
        for (int i = 0; i < numThreads; ++i)
        {
            //std::cout << "main() : creating thread, " << i << std::endl;
            int rc = pthread_create(&threads[i], NULL, RunGenerator, reinterpret_cast< void* >(&threadDatas[i]));
            if (rc)
            {
                std::cout << "Error:unable to create thread," << rc << std::endl;
                exit(-1);
            }
        }

        // free attribute and wait for the other threads
        pthread_attr_destroy(&attr);
        for (int i = 0; i < numThreads; ++i)
        {
            int rc = pthread_join(threads[i], &status);
            if (rc)
            {
                std::cout << "Error:unable to join," << rc << std::endl;
                exit(-1);
            }
            Status< float, double, long double > st = *(reinterpret_cast< Status< float, double, long double >* >(status));
            //std::cout << "Main : completed thread id :" << i;
            //std::cout << " exiting with status :" << st.Size << " " << st.TInc << " " << st.TDec << " " << st.BInc << std::endl;
            if (std::abs(st.TInc - st.TDec) > std::numeric_limits< float >::epsilon())
            {
                double a = std::abs(st.BInc - static_cast< double >(st.TInc));
                double b = std::abs(st.BInc - static_cast< double >(st.TDec));
                //std::cout << st.Size << "\t";
                if (a < b)
                {
                    //std::cout << "inc" << std::endl;
                }
                else if (a > b)
                {
                    //std::cout << "dec" << std::endl;
                }
                else
                {
                    //std::cout << "tie" << std::endl;
                }
                if (std::abs(a - b) < std::numeric_limits< double >::epsilon())
                {
                    result[st.Size][0]++; // 0 for tie
                }
                else if (a < b)
                {
                    result[st.Size][1]++; // 1 for indicating that inc error is smaller
                }
                else if (a > b)
                {
                    result[st.Size][2]++; // 2 for indicating that dec error is smaller
                }
                else
                {
                    result[st.Size][0]++; // 0 for tie
                }
            }
            free(status);
        }
    }

    for (auto it = result.cbegin(); it != result.cend(); ++it)
    {
        for (auto jt = it->second.cbegin(); jt != it->second.cend(); ++jt)
        {
            // (vector size) (tie or inc or dec) (the number of cases)
            std::cout << it->first << "\t" << jt->first << "\t" << jt->second << std::endl;
        }
    }

    //std::cout << "Main: program exiting." << std::endl;
    pthread_exit(NULL);
}

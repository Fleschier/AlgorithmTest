#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include<thread>
#include<mutex>
#include"function_wrapper.h"
#include"threadsafe_queue.h"
#include"join_threads.h"
#include<future>
#include<type_traits>   //包含result_of<>
#include<atomic>
#include<vector>
#include<iostream>


#include <unistd.h>       // for syscall()
#include <sys/syscall.h>  // for SYS_xxx definitions


class thread_pool
{
private:
  // 下面4个成员变量的声明顺序不可以随便更换，否则可能出现部分不能被正确销毁
  std::atomic_bool done;    //原子变量
  threadsafe_queue<function_wrapper> work_queue;    //任务队列
  std::vector<std::thread> threads;     //工作线程
  join_threads joiner;    // 用来等待所有工作线程的结束（保证线程池销毁前所有工作线程已经结束）

  std::condition_variable data_cond;    // 条件等待变量
  std::mutex work_mut;

  void worker_thread();

public:
  thread_pool();
  ~thread_pool();
  void run_pending_task();

  template<typename FunctionType>
  //任务函数f的返回值为std::result_of<FunctionType()>::type
  std::future<typename std::result_of<FunctionType()>::type> submit(FunctionType f){   //可以等待线程任务完成的submit函数
    typedef typename std::result_of<FunctionType()>::type
        result_type;
    std::packaged_task<result_type()> task(std::move(f));   // 将任务函数包装为packaged_task

    std::future<result_type> res(task.get_future());    // 获取task的future共享对象
    work_queue.push(std::move(task));   //提交到任务队列中
    data_cond.notify_one();   //提交任务之后唤醒工作线程
    return res;   // 将任务的返回值（future类型）返回
  }

};

#endif // THREAD_POOL_H

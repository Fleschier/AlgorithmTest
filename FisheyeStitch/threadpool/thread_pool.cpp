#include "thread_pool.h"

thread_pool::thread_pool():
  done(false), joiner(threads){
  //unsigned const thread_count = std::thread::hardware_concurrency();    // 根据电脑硬件获取最大并发数
  unsigned int hardware_threads = std::thread::hardware_concurrency();
  unsigned const thread_count = hardware_threads!=0 ? hardware_threads : 2;

  printf("the computer hardware_concurrency is: %d \n", thread_count);
  try{
    for(unsigned i = 0; i < thread_count; ++i){   //根据最大并发数生成工作线程，并加入到vector<>中
        threads.push_back(
              std::thread(&thread_pool::worker_thread, this));
        //printf("success add a worker_thread, ID: %d\n", threads[i].get_id() );
        std::cout << "success add a worker_thread, ID:" << threads[i].get_id() << std::endl;
      }
  }
  catch(...){
    printf("something wrong!\n");
    done = true;
    throw;
  }
  printf("done!\n");
}

thread_pool::~thread_pool(){
  done = true;
  printf("begin to release threads...\n");
  data_cond.notify_all();
  // make sure that all the worker threads are released!
  for(auto& t : threads){
      if(t.joinable())
        t.join();
  }
  printf("thread pool deleted done!\n");
}

void thread_pool::worker_thread(){
  // modified version
  std::unique_lock<std::mutex> lock(work_mut, std::defer_lock);   // 初始化时不上锁
  while(!done){
      function_wrapper task;

      std::unique_lock<std::mutex> lock(work_mut);
      data_cond.wait(lock, [&task, this]{return (work_queue.try_pop(task) || done);});   // 工作线程等待任务队列来任务
      if(done){
          break;
      }
//      printf("队列待执行任务数： %d\n", work_queue.length());
      if(lock.owns_lock())    // 如果持有锁
        lock.unlock();  // 解锁，然后执行任务
      task();
    }

  // original code
//  while(!done){     //工作线程循环等待
//      function_wrapper task;
//      if(work_queue.try_pop(task)){
//          printf("剩余任务数： %d\n", work_queue.length());
//          task();   // function_wrapper类重载了（）运算符
//        }
//      else{
//          std::this_thread::yield();
//          /*std::this_thread::yield() 的目的是避免一个线程
//           * (that should be used in a case where you are in a busy waiting state)
//           * 频繁与其他线程争抢CPU时间片, 从而导致多线程处理性能下降.
//           * std::this_thread::yield() 是让当前线程让渡出自己的CPU时间片(给其他线程使用)
//           * std::this_thread::sleep_for() 是让当前休眠”指定的一段”时间.
//           * sleep_for()也可以起到 std::this_thread::yield()相似的作用,
//           * (即:当前线程在休眠期间, 自然不会与其他线程争抢CPU时间片)但两者的使用目的是大不相同的:
//           * std::this_thread::yield() 是让线程让渡出自己的CPU时间片(给其他线程使用)
//           * sleep_for() 是线程根据某种需要, 需要等待若干时间.
//           */
//        }
//    }
  std::cout << "worker thread released.\n";
}

void thread_pool::run_pending_task(){    // 执行挂起的任务
  // modified version
  if(work_queue.length() > 1){
      data_cond.notify_all();
    }
  else if(work_queue.empty()){
      return;
    }
  else {
      data_cond.notify_one();
    }


//  // original code
//  function_wrapper task;
//  //printf("剩余任务数： %d\n", work_queue.length());
//  if(work_queue.try_pop(task)){
//      printf("execute pending task...\n");
//      task();
//  }
//  else{
//      std::this_thread::yield();
//  }
}

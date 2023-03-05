#ifndef THREADSAFE_QUEUE_H
#define THREADSAFE_QUEUE_H

#include<thread>
#include<mutex>
#include<memory>
#include<condition_variable>
#include<queue>

template<typename T>
class threadsafe_queue{
private:
  mutable std::mutex mut;   // 互斥量  因为存在互斥量，所以threadsafe_queue本身也是不可复制的！
  std::queue<std::shared_ptr<T>> data_queue;
  std::condition_variable data_cond;  // 条件变量
public:
  threadsafe_queue();
  void wait_and_pop(T& value);
  bool try_pop(T& value);
  std::shared_ptr<T> wait_and_pop();
  std::shared_ptr<T> try_pop();
  void push(T new_value);
  bool empty() const;
  int length() const;
};

// 模板函数的实现一定要与声明在同一个头文件中！

template<typename T>
threadsafe_queue<T>::threadsafe_queue(){}

template<typename T>
void threadsafe_queue<T>::wait_and_pop(T& value){
  std::unique_lock<std::mutex> lock(mut);   // 这里只能用unique_lock，因为lock_guard没有unlock接口。wait的时候，需要对互斥资源解锁。
  // 但是lock_guard比unique_lock性能要好。
  data_cond.wait(lock, [this]{return !data_queue.empty();});
  // 如果wait在被调用的时候等待的条件都满足（lambda函数返回true），则返回，
  //否则线程进入休眠状态，wait会解锁互斥元，直到别的线程调用notify_one或者notify_all来唤醒
  value = std::move(*data_queue.front());   // 右值引用的传递，避免了复制的内存开销。将queue的一个元素复制到value上
  data_queue.pop();

}

template<typename T>
int threadsafe_queue<T>::length() const{
  std::lock_guard<std::mutex> lock(mut);
  return data_queue.size();
}

template<typename T>
bool threadsafe_queue<T>::try_pop(T &value){
  std::lock_guard<std::mutex> lock(mut);
  if(data_queue.empty()){
      return false;
    }
  value = std::move(*data_queue.front());
  data_queue.pop();
  return true;
}

template<typename T>
std::shared_ptr<T> threadsafe_queue<T>::wait_and_pop(){
  /*
   * 和 unique_ptr、weak_ptr 不同之处在于，多个 shared_ptr 智能指针可以共同使用同一块堆内存。
   * 并且，由于该类型智能指针在实现上采用的是引用计数机制，即便有一个 shared_ptr 指针放弃了堆内存的“使用权”
   * （引用计数减 1），也不会影响其他指向同一堆内存的 shared_ptr 指针（只有引用计数为 0 时，堆内存才会被自动释放）
   *
   * 可以通过构造函数、std::make_shared<T>辅助函数和reset方法来初始化shared_ptr
   * 注意，不能将一个原始指针直接赋值给一个智能指针，如下所示，原因是一个是类，一个是指针。
   * std::shared_ptr<int> p4 = new int(1);// error
   * 不要用一个原始指针初始化多个shared_ptr，原因在于，会造成二次销毁
  */
  std::unique_lock<std::mutex> lock(mut);
  data_cond.wait(lock, [this]{return !data_queue.empty();});
  std::shared_ptr<T> res = data_queue.front();
  data_queue.pop();
  return true;
}

template<typename T>
std::shared_ptr<T> threadsafe_queue<T>::try_pop(){
  std::lock_guard<std::mutex> lock(mut);
  if(data_queue.empty()){
      return std::shared_ptr<T>();
    }
  std::shared_ptr<T> res = data_queue.front();
  data_queue.pop();
  return res;
}

template<typename T>
void threadsafe_queue<T>::push(T new_value){
  std::shared_ptr<T> data(
        std::make_shared<T>(std::move(new_value))   // 获取共享指针
        );
  std::lock_guard<std::mutex> lock(mut);
  data_queue.push(data);
  data_cond.notify_one();   // 唤醒一个可能存在的因为队列为空而陷入等待的线程
}

template<typename T>
bool threadsafe_queue<T>::empty() const{
  std::lock_guard<std::mutex> lock(mut);
  return data_queue.empty();
}

#endif // THREADSAFE_QUEUE_H

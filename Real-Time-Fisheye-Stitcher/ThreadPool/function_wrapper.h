#ifndef FUNCTION_WRAPPER_H
#define FUNCTION_WRAPPER_H

#include<memory>

class function_wrapper
{
  struct impl_base{
    virtual void call()=0;    //定义为虚函数，由派生类给出具体实现
    virtual ~impl_base(){}
  };

  std::unique_ptr<impl_base> impl;    //unique_ptr<>是不可复制的，只能被移动

  template<typename F>
  struct impl_type : impl_base{
    F f;
    impl_type(F&& f_): f(std::move(f_)){} // 这里通过std::move()将右值引用f_赋值给f，避免了复制的内存开销
    // 同时，(F&& f_)形参也能接受左值作为函数参数
    void call(){f();}   //重载了()运算符，可以像调用函数一样使用实例化对象： f()
  };

public:
  template<typename F>
  function_wrapper(F&& f): impl(new impl_type<F>(std::move(f))){}

  /*
  第一个()是操作符的名称 - 它是()在对象上使用时调用的操作符。第二个()是参数
  你可以重载()操作符来调用你的对象，就像它是一个函数一样：

  class A {
  public:
      void operator()(int x, int y) {
          // Do something
      }
  };

  A x;
  x(5, 3); // at this point operator () gets called
  */

  void operator()(){impl->call();}
  //void call() { impl->call(); }

  function_wrapper() = default;   //显式地告诉编译器自动生成默认的无参构造函数

  // 下面的两个移动构造函数和三个删除拷贝和赋值运算符，使得该类变为只可移动的类，而不能进行复制
  function_wrapper(function_wrapper&& other): impl(std::move(other.impl)){}   //移动构造函数

  function_wrapper& operator=(function_wrapper&& other){    //重载赋值运算符，支持移动操作
    impl = std::move(other.impl);
    return *this;
  }

  function_wrapper(const function_wrapper&) = delete;  //删除拷贝构造函数

  function_wrapper(function_wrapper&)=delete;   //删除拷贝构造函数

  function_wrapper& operator=(const function_wrapper&)=delete;    //删除拷贝赋值
};

#endif // FUNCTION_WRAPPER_H

// Copyright (c) 2017, Pacific Biosciences of California, Inc.
//
// All rights reserved.
//
// THIS SOFTWARE CONSTITUTES AND EMBODIES PACIFIC BIOSCIENCES' CONFIDENTIAL
// AND PROPRIETARY INFORMATION.
//
// Disclosure, redistribution and use of this software is subject to the
// terms and conditions of the applicable written agreement(s) between you
// and Pacific Biosciences, where "you" refers to you or your company or
// organization, as applicable.  Any other disclosure, redistribution or
// use is prohibited.
//
// THIS SOFTWARE IS PROVIDED BY PACIFIC BIOSCIENCES AND ITS CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
// OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
// ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// File Description:
/// \brief  implementation of a mutex wrapper
//
// Programmer: Mark Lakata


#ifndef APP_COMMON_LOCKED_H
#define APP_COMMON_LOCKED_H

#include <mutex>
#include <type_traits>

#include <pacbio/logging/Logger.h>


namespace PacBio {
namespace Threading {

/// This exception is thrown if the Lockable instance times-out
/// while grabbing its mutex.
struct PBMutexTimeoutException : public std::runtime_error
{
using std::runtime_error::runtime_error;
};

class LockedBase
{
public:
    virtual void Detach() = 0;
    virtual ~LockedBase() {}
};

/// Lockable is meant to be added to any types that need to have a lock mechanism.
/// By deriving your class Foo from this class, the Locked<Foo> class below can
/// be used.
/// By default, the timeout is infinite. By supplying the integer template
/// parameter in milliseconds, the mutex will timeout and throw 
/// PBMutexTimeoutException.
/// The LockMutex and UnlockMutex methods are not intended to be used directly.
template<int timeoutMs=0>
class Lockable
{
public:
    ~Lockable()
    {
        if (owner_) owner_->Detach();
        if (locked_) UnlockMutex();
    }
    /// This locks the object. This method is not to be used directly.
    /// Use the Locked<T> class to create an RAII based lock.
    void LockMutex(LockedBase* owner) const
    {
        if (timeoutMs == 0)
        {
            Mutex().lock();
        }
        else
        {
            ;
            if (!Mutex().try_lock_for(Timeout()))
            {
                throw PacBio::PBExceptionEx<PBMutexTimeoutException>("Locked timed-out",__FILE__,__LINE__, __FUNCTION__, PB_TRACEBACK() );
            }
        }
        locked_ = true;
        owner_ = owner;
    }

    /// Unlocks the object. This method is not to be used directly.
    /// \param force If true, will unlock the mutex regardless if it is locked
    /// or not.
    void UnlockMutex(bool force = false) const
    {
        if (locked_)
        {
            locked_ = false;
            Mutex().unlock();
        }
        else if (!force)
        {
            throw PBException("Attempt to unlock a mutex that was not locked");
        }
        owner_ = nullptr;
    }
private:
    std::timed_mutex& Mutex() const { return mutex_; }
    std::chrono::milliseconds Timeout() const { 
        return std::chrono::milliseconds(timeoutMs);
    }

    mutable std::timed_mutex mutex_;
    mutable bool locked_ = false;
    mutable LockedBase* owner_ = nullptr;
};

/// The Locked class is used to create a pointer whose lifetime dictates the
/// mutex lock. The following example declares a Foo contain that can be locked
/// for a maximum of 10 milliseconds, 10 being the parameter to the Lockable
/// template.
/// \example
///   class Foo : Lockable<10> { int a; int b; };
///   Foo globalFoo; // some global instance that needs to be accessed from more than one thread atomically.
///   {
///     auto foo = Locked<Foo>(&globalFoo); // mutex lock acquired here
///     foo->a = 1;
///     foo->b = 2;
///   } // mutex lock released here
///
/// If the timed_mutex is not unlocked within the timeout, an exception of type
/// PBMutexTimeoutException (derived from std::runtime_error) will be throw.
/// \tparam T : the Type of the object that is to be locked

template<typename T>
class Locked : public LockedBase
{
   //    static_assert(std::is_base_of<Lockable, T>::value,"The templated T must be derived from Lockable");

private:
    Locked(const Locked& x) = delete;

    Locked& operator=(const Locked& other) = delete;

public:
    /// normal constructor. Creates a locked smart pointer that
    /// locks the data model on construction.
    /// Locking a nullptr will throw
    explicit Locked(T* t) : object_(t)
    {
        if (object_)
        {
            object_->LockMutex(this);
            PBLOG_DEBUG << ">>> Locked " << (void*) object_;

        }
        else
        {
            throw PBException("Can not lock a nullptr");
        }
    }

    /// move copy constructor
    Locked(Locked&& x) : object_(nullptr)
    {
        PBLOG_DEBUG << "=== Lock moved " << (void*)object_;
        std::swap(object_, x.object_);
    }

    /// The destructor releases the lock on the underlying model.
    ///
    ~Locked() override
    {
        if (object_)
        {
            object_->UnlockMutex();

            PBLOG_DEBUG << "<<< Unlocked " << (void*) object_;
        }
        else
            {
            PBLOG_DEBUG << "<<< Not Unlocked, nullptr";
        }
    }

    /// get pointer to object. May return nullptr
    T* operator->() { 
        if (!object_) throw PBException("Can't get pointer to nullptr in Locked object");
        return object_; 
    }

    /// get const pointer to object. May return nullptr
    const T* operator->() const {
        if (!object_) throw PBException("Can't get pointer to nullptr in Locked object");
        return object_; 
    }

    /// get reference to object
    T& operator()() {
        if (!object_) throw PBException("Can't dereference nullptr in Locked object");
        return *object_;
    }

    /// get reference to object
    T& operator*() {
        if (!object_) throw PBException("Can't dereference nullptr in Locked object");
        return *object_;
    }

    // get pointer to object
    T* get() { return object_; }

    operator bool() const { return object_ != nullptr; }

    void Detach() override
    {
        object_ = nullptr;
    }

private:
    T* object_ = nullptr;
};

}}

#endif //APP_COMMON_LOCKED_H

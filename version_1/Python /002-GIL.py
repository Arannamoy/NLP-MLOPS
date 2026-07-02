# Global Interpreter Lock

import threading
import time,datetime


def brew_chain():
    print(f"{threading.current_thread().name} started brewing. {datetime.datetime.now().strftime("%H:%M:%S")}")
    count=0
    for i in range(0,1000_000_000):
        count+=1
    print(f"{threading.current_thread().name} finished brewing. {datetime.datetime.now().strftime("%H:%M:%S")}")

brew_thread_1=threading.Thread(target=brew_chain,name="Barista-1")
brew_thread_2=threading.Thread(target=brew_chain,name="Barista-2")

brew_thread_1.start()
brew_thread_2.start()
brew_thread_1.join()
brew_thread_2.join()

print("\nFinished")
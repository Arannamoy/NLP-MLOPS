import threading
import time
import datetime
def take_orders():
    for i in range(1,4):
        print(f"Taking order for #{i}. {datetime.datetime.now().strftime("%H:%M:%S")}")
        time.sleep(5)

def brew_chai():
    for i in range(1,4):
        print(f"brew chai for #{i}. {datetime.datetime.now().strftime("%H:%M:%S")}")
        time.sleep(6)

# creating thread
order_thread=threading.Thread(target=take_orders)
brew_thread=threading.Thread(target=brew_chai)

order_thread.start()
brew_thread.start()

order_thread.join()
brew_thread.join()

print(f"All orders taken and chai brewed.")
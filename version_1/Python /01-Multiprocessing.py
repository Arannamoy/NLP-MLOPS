from multiprocessing import Process
import time
from datetime import datetime

def brew_chai(name):
    print(f"Start of {name} chai brewing. {datetime.now().strftime("%H:%M:%S")}")
    time.sleep(3)
    print(f"End of {name} chai brewing. {datetime.now().strftime("%H:%M:%S")}")

if __name__=="__main__":
    chai_makers=[
        Process(target=brew_chai, args=(f"Chai Makers #{i+1}",))
        for i in range(10)
    ]

    # Start all process 
    for p in chai_makers:
        p.start()
    
    # wait for all process
    for p in chai_makers:
        p.join()

    print(f"All chai served, {datetime.now().strftime("%H:%M:%S")}")

    

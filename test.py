import threading
from time import sleep

def testthread():
    global b
    b = True
    i = 0
    while(i < 10):
        i += 1
        sleep(1)
    b = False
    return 0

b = False
x = threading.Thread(target=testthread)
x.start()
i = 0
while(i < 100):
    print(b)
    sleep(0.5)
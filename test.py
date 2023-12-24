import threading
from time import sleep

def testthread(state):
    print(state)
    state["running"] = True
    i = 0
    while(i < 10):
        i += 1
        sleep(1)
    state["running"] = False
    return 0

state = [1]
state[0] = {"running": False, "xyz": 1}
print(state[0])

for s in state:
    x = threading.Thread(target=testthread, args=(s, ))
    x.start()
i = 0
while(i < 100):
    print(state[0])
    sleep(0.5)
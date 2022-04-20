import pickle
import numpy as np
import matplotlib.pyplot as plt
import time

reward = pickle.load(open("reward-storage.p", "rb"))
avg_rew = []
top = []
mid = []
tq = []
nipe = []
nifi = []
nini = []
null = []
worst = []
fifo =[]

summe = 0

smoothing = 1000
p = 5  # produkte
m = 5  # maschinen
s = p*m # schritte

for x in range(len(reward)):
    if x < smoothing:
        avg = sum(reward[0:(x + 1)]) / (x + 1)
        avg_rew.append(avg)

    else:
        avg = sum(reward[x - smoothing:x]) / smoothing
        avg_rew.append(avg)

    summe += reward[x]

    top.append(s ** 3)
    mid.append((s/2) ** 3)
    tq.append((s / 4 * 3) ** 3)
    nipe.append((s * 0.9) ** 3)
    nifi.append((s * 0.95) ** 3)
    nini.append((s * 0.99) ** 3)
    worst.append(-s/2*90)
    null.append(0)
    fifo.append(5200)

print(summe/len(reward))
print((summe/len(reward))**0.3333)

plt.figure().set_size_inches(21, 9)
plt.plot(reward)
plt.plot(avg_rew, c="red", label='Average reward')
plt.plot(top, c="green", label='100% complete')
plt.plot(nifi, c="brown", label='95% complete')
plt.plot(nini, c="orange", label='99% complete')
plt.plot(nipe, c="pink", label='90% complete')
plt.plot(tq, c="orange", label='75% complete')
plt.plot(mid, c="yellow", label='50% complete')
plt.plot(null, c="Black", label='0% complete')
plt.plot(fifo, c="Black", label='FIFO')
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.legend(loc='upper left', frameon=True)
plt.savefig('update.pdf', dpi=5000, transparent=True, bbox_inches='tight')
plt.savefig("update.jpg", dpi=150)
plt.show()

# 50 -- 2_500 ## bei 6*6
# 75 -- 7_500
# 90 -- 15_000
# 95 -- 20_000
# 99 -- 50_000

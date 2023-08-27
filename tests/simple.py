"""
Test objects from base.py
"""

# %% Imports and settings
import stisim as ss
import matplotlib.pyplot as plt


ppl = ss.People(10000)
ppl.networks = ss.ndict(ss.simple_sexual(), ss.maternal())

hiv = ss.HIV()
hiv.pars['beta'] = {'simple_sexual': [0.0008, 0.0004], 'maternal': [0.2, 0]}

sim = ss.Sim(people=ppl, modules=[hiv, ss.Pregnancy()], n_years=1000)
sim.initialize()
sim.run()

plt.figure()
plt.plot(sim.tivec, sim.results.hiv.n_infected)
plt.title('HIV number of infections')


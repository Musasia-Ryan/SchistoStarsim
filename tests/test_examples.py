"""
Run simplest tests
"""

# %% Imports and settings
import stisim as ss
import matplotlib.pyplot as plt

def test_sir():
    ppl = ss.People(10000)
    ppl.networks = ss.ndict(ss.RandomNetwork(n_contacts=ss.poisson(5)))
    sir = ss.SIR()
    sim = ss.Sim(people=ppl, diseases=sir)
    sim.run()

    plt.figure()
    plt.stackplot(
        sim.tivec,
        sir.results.n_susceptible,
        sir.results.n_infected,
        sir.results.n_recovered,
        sim.results.new_deaths.cumsum(),
    )
    plt.legend(['Susceptible', 'Infected', 'Recovered', 'Dead'])


def test_ncd():
    ppl = ss.People(10000)
    ppl.networks = None
    ncd = ss.NCD()
    sim = ss.Sim(people=ppl, diseases=ncd)
    sim.run()

    plt.figure()
    plt.stackplot(
        sim.tivec,
        ncd.results.n_not_at_risk,
        ncd.results.n_at_risk-ncd.results.n_affected,
        ncd.results.n_affected,
        sim.results.new_deaths.cumsum(),
    )
    plt.legend(['Not at risk','At risk','Affected', 'Dead'])


if __name__ == '__main__':
    sim1 = test_sir()
    sim2 = test_ncd()

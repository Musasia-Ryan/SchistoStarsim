"""
Define schisto model.
Based on cholera
"""

import numpy as np
import starsim as ss 
import pylab as pl
import sciris as sc


class Schistosomiasis(ss.Infection):
    """
    Schisto
    """
    def __init__(self, pars=None, par_dists=None, *args, **kwargs):
        """ Initialize with parameters """

        pars = ss.omergeleft(pars,
            # Natural history parameters, all specified in days
            # Initial conditions and beta
            init_prev = 0.18,
            beta = None,

            # Environmental parameters
            beta_env = 0.5 / 3,  # Scaling factor for transmission from environment,
            half_sat_rate = 1000000,   # Infectious dose in water sufficient to produce infection in 50% of  exposed, from Mukandavire et al. (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3102413/)
            shedding_rate = 10,    # Rate at which infectious people shed bacteria to the environment (per day), from Mukandavire et al. (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3102413/)
            decay_rate = 0.0625,    # Rate at which bacteria in the environment dies (per day), from Chao et al. and Mukandavire et al. citing https://pubmed.ncbi.nlm.nih.gov/8882180/
            p_env_transmit = 0,    # Probability of environmental transmission - filled out later
        )

        par_dists = ss.omergeleft(par_dists,
            init_prev      = ss.bernoulli,
            p_env_transmit = ss.bernoulli,
        )

        super().__init__(pars=pars, par_dists=par_dists, *args, **kwargs)

        return

    def init_results(self, sim):
        """
        Initialize results
        """
        super().init_results(sim)
        self.results += [
            ss.Result(self.name, 'env_prev', sim.npts, dtype=float),
            ss.Result(self.name, 'env_conc', sim.npts, dtype=float),
        ]
        return

    def update_pre(self, sim):
        """
        Update today's environmental prevalence
        """
        self.calc_environmental_prev(sim)

    def calc_environmental_prev(self, sim):
        """
        Calculate environmental prevalence
        """
        p = self.pars
        r = self.results 

        n_infected = len(ss.true(self.infectious))
        old_prev = self.results.env_prev[sim.ti-1]

        new_bacteria = p.shedding_rate * n_infected
        old_bacteria = old_prev * (1 - p.decay_rate)

        r.env_prev[sim.ti] = new_bacteria + old_bacteria
        r.env_conc[sim.ti] = r.env_prev[sim.ti] / (r.env_prev[sim.ti] + p.half_sat_rate)

    def set_prognoses(self, sim, uids, source_uids=None):
        """ Set prognoses for those who get infected """
        super().set_prognoses(sim, uids, source_uids)

        self.susceptible[uids] = False
        self.infected[uids] = True
        self.ti_infected[uids] = sim.ti

        p = self.pars

        return

    def make_new_cases(self, sim):
        """ Add indirect transmission """

        pars = self.pars
        res = self.results

        # Make new cases via direct transmission
        super().make_new_cases(sim)

        # Make new cases via indirect transmission
        p_transmit = res.env_conc[sim.ti] * pars.beta_env
        pars.p_env_transmit.set(p=p_transmit)
        new_cases = pars.p_env_transmit.filter(sim.people.uid[self.susceptible]) # TODO: make syntax nicer
        if new_cases.any():
            self.set_prognoses(sim, new_cases, source_uids=None)
        return

    def update_death(self, sim, uids):
        """ Reset infected/recovered flags for dead agents """
        for state in ['susceptible', 'infected']:
            self.statesdict[state][uids] = False
        return

    def update_results(self, sim):
        super().update_results(sim)
        res = self.results
        ti = sim.ti
        res.prevalence[ti]     = res.n_infected[ti] / np.count_nonzero(sim.people.alive)
        res.new_infections[ti] = np.count_nonzero(self.ti_infected == ti)
        res.cum_infections[ti] = np.sum(res.new_infections[:ti+1])
        return


class Treatment(ss.Intervention):

    def __init__(self, prob=0.5, years=None):
        super().__init__()
        self.prob = prob
        self.years = years

    def apply(self, sim, *args, **kwargs):
        if sim.yearvec[sim.ti] in self.years:
            schisto = sim.diseases.schistosomiasis
            eligible_ids = sim.people.age > 5
            n_eligible = len(eligible_ids)
            is_treated = np.random.rand(n_eligible) < self.prob
            treat_ids = eligible_ids[is_treated]

            schisto.infected[treat_ids] = False
            schisto.susceptible[treat_ids] = True


# Function to run sim
def run_schisto(n_agents = 5000, treatment_years = None):

    pars = dict(
        start = 2019,
        n_years = 10,
        n_agents = n_agents,
        networks = 'random',
    )

    # Make the disease
    schisto = Schistosomiasis()

    # Make the treatment intervention
    MDA = Treatment(prob=1, years = treatment_years)
    sim = ss.Sim(pars, diseases=schisto, interventions=MDA)
    sim.run()
    return sim


if __name__ == '__main__':

    treatment_years = [2019, 2020, 2021]
    sim = run_schisto(treatment_years=treatment_years)

    # Pull out results
    results = sim.results.schistosomiasis

    pl.figure()
    pl.subplot(2,1,1)
    pl.title('Number infected')
    pl.plot(sim.yearvec, results.n_infected)
    for ty in treatment_years:
        pl.axvline(ty, ls='--', c='k')
    pl.subplot(2,1,2)
    pl.title('Prevalence')
    pl.plot(sim.yearvec, results.prevalence)
    for ty in treatment_years:
        pl.axvline(ty, ls='--', c='k')
    sc.figlayout()
    pl.show()


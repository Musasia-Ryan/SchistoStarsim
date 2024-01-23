'''
Networks that connect people within a population
'''

# %% Imports
import numpy as np
import sciris as sc
import starsim as ss
import scipy.optimize as spo
import scipy.stats as sps
import scipy.spatial as spsp
import pandas as pd
from scipy.stats._distn_infrastructure import rv_frozen
import scipy.stats as sps


# Specify all externally visible functions this file defines
__all__ = ['Networks', 'Network', 'NetworkConnector', 'SexualNetwork', 'mf', 'msm', 'mf_msm', 'hpv_network', 'maternal', 'embedding', 'static']


class Network(ss.Module):
    """
    A class holding a single network of contact edges (connections) between people
    as well as methods for updating these.

    The input is typically arrays including: person 1 of the connection, person 2 of
    the connection, the weight of the connection, the duration and start/end times of
    the connection.

    Args:
        p1 (array): an array of length N, the number of connections in the network, with the indices of people
                   on one side of the connection.
        p2 (array): an array of length N, the number of connections in the network, with the indices of people
                    on the other side of the connection.
        beta (array): an array representing relative transmissibility of each connection for this network - TODO, do we need this?
        label (str): the name of the network (optional)
        kwargs (dict): other keys copied directly into the network

    Note that all arguments (except for label) must be arrays of the same length,
    although not all have to be supplied at the time of creation (they must all
    be the same at the time of initialization, though, or else validation will fail).

    **Examples**::

        # Generate an average of 10 contacts for 1000 people
        n_contacts_pp = 10
        n_people = 1000
        n = n_contacts_pp * n_people
        p1 = np.random.randint(n_people, size=n)
        p2 = np.random.randint(n_people, size=n)
        beta = np.ones(n)
        network = ss.Network(p1=p1, p2=p2, beta=beta, label='rand')
        network = ss.Network(dict(p1=p1, p2=p2, beta=beta), label='rand') # Alternate method

        # Convert one network to another with extra columns
        index = np.arange(n)
        self_conn = p1 == p2
        network2 = ss.Network(**network, index=index, self_conn=self_conn, label=network.label)
    """

    def __init__(self, pars=None, key_dict=None, vertical=False, *args, **kwargs):

        # Initialize as a module
        super().__init__(pars, *args, **kwargs)

        # Each relationship is characterized by these default set of keys, plus any user- or network-supplied ones
        default_keys = {
            'p1': ss.int_,
            'p2': ss.int_,
            'beta': ss.float_,
        }
        self.meta = ss.omerge(default_keys, key_dict)
        self.vertical = vertical  # Whether transmission is bidirectional

        # Initialize the keys of the network
        self.contacts = sc.objdict()
        for key, dtype in self.meta.items():
            self.contacts[key] = np.empty((0,), dtype=dtype)

        # Set data, if provided
        for key, value in kwargs.items():
            self.contacts[key] = np.array(value, dtype=self.meta.get(key))
            self.initialized = True

        # Define states using placeholder values
        self.participant = ss.State('participant', bool, fill_value=False)
        self.debut = ss.State('debut', float, fill_value=0)
        return


    def initialize(self, sim):
        super().initialize(sim)
        return

    def __len__(self):
        try:
            return len(self.contacts.p1)
        except:  # pragma: no cover
            return 0

    def __repr__(self, **kwargs):
        """ Convert to a dataframe for printing """
        namestr = self.name
        labelstr = f'"{self.label}"' if self.label else '<no label>'
        keys_str = ', '.join(self.contacts.keys())
        output = f'{namestr}({labelstr}, {keys_str})\n'  # e.g. Network("r", p1, p2, beta)
        output += self.to_df().__repr__()
        return output

    def __contains__(self, item):
        """
        Check if a person is present in a network

        Args:
            item: Person index

        Returns: True if person index appears in any interactions
        """
        return (item in self.contacts.p1) or (item in self.contacts.p2)

    @property
    def members(self):
        """ Return sorted array of all members """
        return np.unique([self.contacts.p1, self.contacts.p2])

    def meta_keys(self):
        """ Return the keys for the network's meta information """
        return self.meta.keys()

    def set_network_states(self, people):
        """
        Many network states depend on properties of people -- e.g. MSM depends on being male,
        age of debut varies by sex and over time, and participation rates vary by age.
        Each time states are dynamically grown, this function should be called to set the network
        states that depend on other states.
        """
        pass

    def validate(self, force=True):
        """
        Check the integrity of the network: right types, right lengths.

        If dtype is incorrect, try to convert automatically; if length is incorrect,
        do not.
        """
        n = len(self.contacts.p1)
        for key, dtype in self.meta.items():
            if dtype:
                actual = self.contacts[key].dtype
                expected = dtype
                if actual != expected:
                    self.contacts[key] = np.array(self.contacts[key], dtype=expected)  # Try to convert to correct type
            actual_n = len(self.contacts[key])
            if n != actual_n:
                errormsg = f'Expecting length {n} for network key "{key}"; got {actual_n}'  # Report length mismatches
                raise TypeError(errormsg)
        return

    def get_inds(self, inds, remove=False):
        """
        Get the specified indices from the edgelist and return them as a dict.
        Args:
            inds (int, array, slice): the indices to find
            remove (bool): whether to remove the indices
        """
        output = {}
        for key in self.meta_keys():
            output[key] = self.contacts[key][inds]  # Copy to the output object
            if remove:
                self.contacts[key] = np.delete(self.contacts[key], inds)  # Remove from the original
        return output

    def pop_inds(self, inds):
        """
        "Pop" the specified indices from the edgelist and return them as a dict.
        Returns arguments in the right format to be used with network.append().

        Args:
            inds (int, array, slice): the indices to be removed
        """
        popped_inds = self.get_inds(inds, remove=True)
        return popped_inds

    def append(self, contacts):
        """
        Append contacts to the current network.

        Args:
            contacts (dict): a dictionary of arrays with keys p1,p2,beta, as returned from network.pop_inds()
        """
        for key in self.meta_keys():
            new_arr = contacts[key]
            n_curr = len(self.contacts[key])  # Current number of contacts
            n_new = len(new_arr)  # New contacts to add
            n_total = n_curr + n_new  # New size
            self.contacts[key] = np.resize(self.contacts[key], n_total)  # Resize to make room, preserving dtype
            self.contacts[key][n_curr:] = new_arr  # Copy contacts into the network
        return

    def to_dict(self):
        """ Convert to dictionary """
        d = {k: self.contacts[k] for k in self.meta_keys()}
        return d

    def to_df(self):
        """ Convert to dataframe """
        df = sc.dataframe.from_dict(self.to_dict())
        return df

    def from_df(self, df, keys=None):
        """ Convert from a dataframe """
        if keys is None:
            keys = self.meta_keys()
        for key in keys:
            self.contacts[key] = df[key].to_numpy()
        return self

    def find_contacts(self, inds, as_array=True):
        """
        Find all contacts of the specified people

        For some purposes (e.g. contact tracing) it's necessary to find all the contacts
        associated with a subset of the people in this network. Since contacts are bidirectional
        it's necessary to check both p1 and p2 for the target indices. The return type is a Set
        so that there is no duplication of indices (otherwise if the Network has explicit
        symmetric interactions, they could appear multiple times). This is also for performance so
        that the calling code doesn't need to perform its own unique() operation. Note that
        this cannot be used for cases where multiple connections count differently than a single
        infection, e.g. exposure risk.

        Args:
            inds (array): indices of people whose contacts to return
            as_array (bool): if true, return as sorted array (otherwise, return as unsorted set)

        Returns:
            contact_inds (array): a set of indices for pairing partners

        Example: If there were a network with
        - p1 = [1,2,3,4]
        - p2 = [2,3,1,4]
        Then find_contacts([1,3]) would return {1,2,3}
        """

        # Check types
        if not isinstance(inds, np.ndarray):
            inds = sc.promotetoarray(inds)
        if inds.dtype != np.int64:  # pragma: no cover # This is int64 since indices often come from utils.true(), which returns int64
            inds = np.array(inds, dtype=np.int64)

        # Find the contacts
        contact_inds = ss.find_contacts(self.contacts.p1, self.contacts.p2, inds)
        if as_array:
            contact_inds = np.fromiter(contact_inds, dtype=ss.int_)
            contact_inds.sort()

        return contact_inds

    def add_pairs(self):
        """ Define how pairs of people are formed """
        pass

    def update(self, people):
        """ Define how pairs/connections evolve (in time) """
        pass

    def remove_uids(self, uids):
        """
        Remove interactions involving specified UIDs
        This method is typically called via `People.remove()` and
        is specifically used when removing agents from the simulation.
        """
        keep = ~(np.isin(self.contacts.p1, uids) | np.isin(self.contacts.p2, uids))
        for k in self.meta_keys():
            self.contacts[k] = self.contacts[k][keep]


class DynamicNetwork(Network):
    def __init__(self, pars=None, key_dict=None):
        key_dict = ss.omerge({'dur': ss.float_}, key_dict)
        super().__init__(pars, key_dict=key_dict)

    def end_pairs(self, people):
        dt = people.dt
        self.contacts.dur = self.contacts.dur - dt

        # Non-alive agents are removed
        active = (self.contacts.dur > 0) & people.alive[self.contacts.p1] & people.alive[self.contacts.p2]
        self.contacts.p1 = self.contacts.p1[active]
        self.contacts.p2 = self.contacts.p2[active]
        self.contacts.beta = self.contacts.beta[active]
        self.contacts.dur = self.contacts.dur[active]


class Networks(ss.ndict):
    def __init__(self, *args, type=Network, connectors=None, **kwargs):
        self.setattribute('_connectors', ss.ndict(connectors))
        super().__init__(*args, type=type, **kwargs)
        return

    def initialize(self, sim):
        for nw in self.values():
            nw.initialize(sim)
        for cn in self._connectors.values():
            cn.initialize(sim)
        return

    def update(self, people):
        for nw in self.values():
            nw.update(people)
        for cn in self._connectors.values():
            cn.update(people)
        return


class SexualNetwork(Network):
    """ Base class for all sexual networks """
    def __init__(self, pars=None):
        super().__init__(pars)

    def active(self, people):
        # Exclude people who are not alive
        return self.participant & (people.age > self.debut) & people.alive

    def available(self, people, sex):
        # Currently assumes unpartnered people are available
        # Could modify this to account for concurrency
        # This property could also be overwritten by a NetworkConnector
        # which could incorporate information about membership in other
        # contact networks
        return np.setdiff1d(ss.true(people[sex] & self.active(people)), self.members) # ss.true instead of people.uid[]?


class mf(SexualNetwork, DynamicNetwork):
    """
    This network is built by **randomly pairing** males and female with variable
    relationship durations.
    """

    def __init__(self, pars=None, key_dict=None):
        pars = ss.omerge({
            'duration_dist': ss.lognorm(mean=15, stdev=15), # Can vary by age, year, and individual pair. Set scale=exp(mu) and s=sigma where mu,sigma are of the underlying normal distribution.
            'participation_dist': sps.bernoulli(p=0.9),  # Probability of participating in this network - can vary by individual properties (age, sex, ...) using callable parameter values
            'debut_dist': sps.norm(loc=16, scale=2),  # Age of debut can vary by using callable parameter values
            'rel_part_rates': 1.0,
        }, pars)

        DynamicNetwork.__init__(self, key_dict)
        SexualNetwork.__init__(self, pars)
        return

    def initialize(self, sim):
        super().initialize(sim)
        self.set_network_states(sim.people)
        self.add_pairs(sim.people, ti=0)
        return

    def set_network_states(self, people, upper_age=None):
        """ Set network states including age of entry into network and participation rates """
        self.set_debut(people, upper_age=upper_age)
        self.set_participation(people, upper_age=upper_age)

    def set_participation(self, people, upper_age=None):
        # Set people who will participate in the network at some point
        year = people.year
        if upper_age is None: uids = people.uid
        else: uids = people.uid[(people.age < upper_age)]
        self.participant[uids] = self.pars.participation_dist.rvs(uids)

    def set_debut(self, people, upper_age=None):
        # Set debut age
        if upper_age is None: uids = people.uid
        else: uids = people.uid[(people.age < upper_age)]
        self.debut[uids] = self.pars.debut_dist.rvs(uids)
        uids_to_update = uids[np.isnan(people.debut[uids])]
        people.debut[uids_to_update] = self.debut[uids_to_update]
        return

    def add_pairs(self, people, ti=None):
        available_m = self.available(people, 'male')
        available_f = self.available(people, 'female')

        # random.choice is not common-random-number safe, and therefore we do
        # not try to Stream-ify the following draws at this time.
        if len(available_m) <= len(available_f):
            p1 = available_m
            p2 = np.random.choice(a=available_f, size=len(p1), replace=False)
        else:
            p2 = available_f
            p1 = np.random.choice(a=available_m, size=len(p2), replace=False)

        beta = np.ones_like(p1)

        # Figure out durations
        # print('DJK TODO')
        if ss.options.multirng and (len(p1) == len(np.unique(p1))):
            # No duplicates and user has enabled multirng, so use slotting based on p1
            dur_vals = self.pars.duration_dist.rvs(p1)
        else:
            dur_vals = self.pars.duration_dist.rvs(len(p1))  # Just use len(p1) to say how many draws are needed

        self.contacts.p1 = np.concatenate([self.contacts.p1, p1])
        self.contacts.p2 = np.concatenate([self.contacts.p2, p2])
        self.contacts.beta = np.concatenate([self.contacts.beta, beta])
        self.contacts.dur = np.concatenate([self.contacts.dur, dur_vals])
        return len(p1)

    def update(self, people, dt=None):
        self.end_pairs(people)
        self.set_network_states(people, upper_age=people.dt)
        self.add_pairs(people)
        return


class msm(SexualNetwork, DynamicNetwork):
    """
    A network that randomly pairs males
    """

    def __init__(self, pars=None):
        default_pars = {
            'part_rates': 0.1,  # Participation rates - can vary by sex and year
            'rel_part_rates': 1.0,
        }

        desired_mean = 5
        desired_std = 3
        mu = np.log(desired_mean**2 / np.sqrt(desired_mean**2 + desired_std**2))
        sigma = np.sqrt(np.log(1 + desired_std**2 / desired_mean**2))
        default_pars['duration_dist'] = sps.lognorm(s=sigma, scale=np.exp(mu)) # Can vary by age, year, and individual pair. Set scale=exp(mu) and s=sigma where mu,sigma are of the underlying normal distribution.

        desired_mean = 18
        desired_std = 2
        mu = np.log(desired_mean**2 / np.sqrt(desired_mean**2 + desired_std**2))
        sigma = np.sqrt(np.log(1 + desired_std**2 / desired_mean**2))
        default_pars['debut_dist'] = sps.lognorm(s=sigma, scale=np.exp(mu))

        pars = ss.omerge(default_pars, pars)
        DynamicNetwork.__init__(self)
        SexualNetwork.__init__(self, pars)

        return

    def initialize(self, sim):
        # Add more here in line with MF network, e.g. age of debut
        # Or if too much replication then perhaps both these networks
        # should be subclasss of a specific network type (ask LY/DK)
        super().initialize(sim)
        self.set_network_states(sim.people)
        self.add_pairs(sim.people, ti=0)
        return

    def set_network_states(self, people, upper_age=None):
        """ Set network states including age of entry into network and participation rates """
        if upper_age is None: uids = people.uid[people.male]
        else: uids = people.uid[people.male & (people.age < upper_age)]

        # Participation
        self.participant[people.female] = False
        pr = self.pars.part_rates
        dist = sps.bernoulli.rvs(p=pr, size=len(uids))
        self.participant[uids] = dist

        # Debut
        self.debut[uids] = self.pars.debut_dist.rvs(len(uids)) # Just pass len(uids) as this network is not crn safe anyway
        uids_to_update = uids[np.isnan(people.debut[uids])]
        people.debut[uids_to_update] = self.debut[uids_to_update]
        return

    def add_pairs(self, people, ti=None):
        # Pair all unpartnered MSM
        available_m = self.available(people, 'm')
        n_pairs = int(len(available_m)/2)
        p1 = available_m[:n_pairs]
        p2 = available_m[n_pairs:n_pairs*2]

        # Figure out durations
        # print('DJK TODO')
        if ss.options.multirng and (len(p1) == len(np.unique(p1))):
            # No duplicates and user has enabled multirng, so use slotting based on p1
            dur = self.pars['duration_dist'].rvs(p1)
        else:
            dur = self.pars['duration_dist'].rvs(len(p1)) # Just use len(p1) to say how many draws are needed

        self.contacts.p1 = np.concatenate([self.contacts.p1, p1])
        self.contacts.p2 = np.concatenate([self.contacts.p2, p2])
        self.contacts.beta = np.concatenate([self.contacts.beta, np.ones_like(p1)])
        self.contacts.dur = np.concatenate([self.contacts.dur, dur])
        return len(p1)

    def update(self, people, dt=None):
        self.end_pairs(people)
        self.set_network_states(people, upper_age=people.dt)
        self.add_pairs(people)
        return


class embedding(mf):
    """
    Heterosexual age-assortative network based on a one-dimensional embedding. Could be made more generic.
    """

    def __init__(self, pars=None, **kwargs):
        """
        Create a sexual network from a 1D embedding based on age

        male_shift is the average age that males are older than females in partnerships
        std is the standard deviation of noise added to the age of each individual when seeking a pair. Larger values will created more diversity in age gaps.
        
        """
        pars = ss.omerge({
            'embedding_func': sps.norm(loc=self.embedding_loc, scale=2),
        }, pars)
        super().__init__(pars, **kwargs)
        return

    @staticmethod
    def embedding_loc(self, sim, uids):
        loc = sim.people.age[uids].values
        loc[sim.people.female[uids]] += 5 # Shift females so they will be paired with older men
        return loc

    def add_pairs(self, people, ti=None):
        available_m = self.available(people, 'male')
        available_f = self.available(people, 'female')

        if not len(available_m) or not len(available_f):
            if ss.options.verbose > 1:
                print('No pairs to add')
            return 0

        available = np.concatenate((available_m, available_f))
        loc = self.pars['embedding_func'].rvs(available)
        loc_f = loc[people.female[available]]
        loc_m = loc[~people.female[available]]

        dist_mat = spsp.distance_matrix(loc_m[:, np.newaxis], loc_f[:, np.newaxis])

        ind_m, ind_f = spo.linear_sum_assignment(dist_mat)
        # loc_f[ind_f[0]] is close to loc_m[ind_m[0]]

        n_pairs = len(ind_f)

        beta = np.ones(n_pairs)

        # Figure out durations
        p1 = available_m[ind_m]
        dur_vals = self.pars['duration_dist'].rvs(p1)

        self.contacts.p1 = np.concatenate([self.contacts.p1, p1])
        self.contacts.p2 = np.concatenate([self.contacts.p2, available_f[ind_f]])
        self.contacts.beta = np.concatenate([self.contacts.beta, beta])
        self.contacts.dur = np.concatenate([self.contacts.dur, dur_vals])
        return len(beta)


class NetworkConnector(ss.Module):
    """
    Template for a connector between networks.
    """
    def __init__(self, *args, networks=None, pars=None, **kwargs):
        super().__init__(pars, requires=networks, *args, **kwargs)
        return

    def set_participation(self, people, upper_age=None):
        pass

    def update(self, people):
        pass


class mf_msm(NetworkConnector):
    """ Combines the MF and MSM networks """
    def __init__(self, pars=None):
        networks = [ss.mf, ss.msm]
        pars = ss.omerge({
            'prop_bi': 0.5,  # Could vary over time -- but not by age or sex or individual
        }, pars)
        super().__init__(networks=networks, pars=pars)

        self.bi_dist = sps.bernoulli(p=self.pars.prop_bi)
        return

    def initialize(self, sim):
        super().initialize(sim)
        self.set_participation(sim.people)
        return

    def set_participation(self, people, upper_age=None):
        if upper_age is None:
            uids = people.uid
        else:
            uids = people.uid[(people.age < upper_age)]
        uids = ss.true(people.male[uids])

        # Get networks and overwrite default participation
        mf = people.networks['mf']
        msm = people.networks['msm']
        mf.participant[uids] = False
        msm.participant[uids] = False

        # Male participation rate uses info about cross-network participation.
        # First, we determine who's participating in the MSM network
        pr = msm.pars.part_rates
        dist = sps.bernoulli.rvs(p=pr, size=len(uids))
        msm.participant[uids] = dist

        # Now we take the MSM participants and determine which are also in the MF network
        msm_uids = ss.true(msm.participant[uids])  # Males in the MSM network
        bi_uids = self.bi_dist.filter(msm_uids)  # Males in both MSM and MF networks
        mf_excl_set = np.setdiff1d(uids, msm_uids)  # Set of males who aren't in the MSM network

        # What remaining share to we need?
        mf_df = mf.pars.part_rates.loc[mf.pars.part_rates.sex == 'm']  # Male participation in the MF network
        mf_pr = np.interp(people.year, mf_df['year'], mf_df['part_rates']) * mf.pars.rel_part_rates
        remaining_pr = max(mf_pr*len(uids)-len(bi_uids), 0)/len(mf_excl_set)

        # Don't love the following new syntax:
        mf_excl_uids = mf_excl_set[sps.uniform.rvs(size=len(mf_excl_set)) < remaining_pr]

        mf.participant[bi_uids] = True
        mf.participant[mf_excl_uids] = True
        return

    def update(self, people):
        self.set_participation(people, upper_age=people.dt)
        return


class hpv_network(mf):
    def __init__(self, pars=None):

        key_dict = {
            'acts': ss.float_,
            'start': ss.float_,
        }

        # Define default parameters
        default_pars = dict()
        default_pars['cross_layer']   = 0.05  # Proportion of agents who have concurrent cross-layer relationships
        default_pars['partner_dist']  = sps.poisson(mu=0.01)  # The number of concurrent sexual partners

        # TODO: Wrap so user can provide mean and dispersion directly - see #168
        mu = 80 # Mean
        alpha = 40 # Dispersion
        sigma2 = mu + alpha * mu**2
        n = mu**2 / (sigma2 - mu)
        p = mu / sigma2
        default_pars['act_dist']      = sps.nbinom(n=n, p=p)  # The number of sexual acts per year

        default_pars['age_act_pars']  = dict(peak=30, retirement=100, debut_ratio=0.5,
                                         retirement_ratio=0.1)  # Parameters describing changes in coital frequency over agent lifespans
        default_pars['condoms']       = 0.2  # The proportion of acts in which condoms are used

        low = 0
        loc = 1
        scale = 1
        a = (low - loc) / scale
        default_pars['duration_dist'] = sps.truncnorm(a=a, b=np.inf, loc=loc, scale=scale)  # Duration of partnerships

        #default_pars['participation'] = None  # Incidence of partnership formation by age
        default_pars['mixing']        = None  # Mixing matrices for storing age differences in partnerships

        self.agebins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
        # Share of females of each age newly having casual relationships
        self.f_participation = [0, 0, 0.10, 0.7, 0.8, 0.6, 0.6, 0.4, 0.1, 0.05, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
        # Share of males of each age newly having casual relationships
        self.m_participation = [0, 0, 0.05, 0.7, 0.8, 0.6, 0.6, 0.4, 0.4, 0.3, 0.1, 0.05, 0.01, 0.01, 0.001, 0.001]

        default_pars['participation_dist'] = sps.bernoulli(p=self.participation)

        pars = ss.omerge(default_pars, pars)
        super().__init__(pars, key_dict)
 
        self.get_layer_probs()
        self.validate_pars()

        return

    def initialize(self, sim):
        super().initialize(sim)
        return self.add_pairs(sim.people, ti=0)

    def validate_pars(self):
        for par in ['partner_dist', 'act_dist', 'duration_dist', 'participation_dist']:
            if not isinstance(self.pars[par], rv_frozen):
                raise Exception(f'Network parameter {par} must be an instance of a scipy frozen distribution')

    def update_pars(self, pars):
        if pars is not None:
            for k, v in pars.items():
                self.pars[k] = v
        return

    @staticmethod
    def participation(self, sim, uids):
        p = np.ones_like(uids, dtype=ss.float_)
        fem = sim.people.female[uids]
        p[fem] = np.interp(sim.people.age[uids[fem]], self.agebins, self.f_participation)
        p[~fem] = np.interp(sim.people.age[uids[~fem]], self.agebins, self.m_participation)
        return p

    def get_layer_probs(self):

        defaults = {}
        mixing = np.array([
            #       0,  5,  10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [10, 0, 0, .1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [15, 0, 0, .1, .1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [20, 0, 0, .1, .1, .1, .1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [25, 0, 0, .5, .1, .5, .1, .1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [30, 0, 0, 1, .5, .5, .5, .5, .1, 0, 0, 0, 0, 0, 0, 0, 0],
            [35, 0, 0, .5, 1, 1, .5, 1, 1, .5, 0, 0, 0, 0, 0, 0, 0],
            [40, 0, 0, 0, .5, 1, 1, 1, 1, 1, .5, 0, 0, 0, 0, 0, 0],
            [45, 0, 0, 0, 0, .1, 1, 1, 2, 1, 1, .5, 0, 0, 0, 0, 0],
            [50, 0, 0, 0, 0, 0, .1, 1, 1, 1, 1, 2, .5, 0, 0, 0, 0],
            [55, 0, 0, 0, 0, 0, 0, .1, 1, 1, 1, 1, 2, .5, 0, 0, 0],
            [60, 0, 0, 0, 0, 0, 0, 0, .1, .5, 1, 1, 1, 2, .5, 0, 0],
            [65, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, .5, 0],
            [70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, .5],
            [75, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        ])

        defaults['mixing'] = mixing

        for pkey, pval in defaults.items():
            if self.pars[pkey] is None:
                self.pars[pkey] = pval

        return

    def set_participation(self, people, upper_age=None):
        if upper_age is None:
            uids = people.uid
        else:
            uids = people.uid[(people.age < upper_age)]

        # Compute number of partners
        f_partnered_inds, f_partnered_counts = np.unique(self.contacts.p1, return_counts=True)
        m_partnered_inds, m_partnered_counts = np.unique(self.contacts.p2, return_counts=True)
        current_partners = np.zeros((len(people)))
        current_partners[f_partnered_inds] = f_partnered_counts
        current_partners[m_partnered_inds] = m_partnered_counts
        partners = self.pars.partner_dist.rvs(len(people)) + 1
        underpartnered = current_partners < partners  # Indices of underpartnered people

        # Set people who will participate in the network at some point
        can_participate = ss.true(people.active * underpartnered)
        self.participant[uids] = self.pars.participation_dist.rvs(can_participate)
        return

    def add_pairs(self, people, ti=0):
        participating = ss.true(self.participant) # Will be the same people each time, with participation decided once per person
        f = participating[people.female[participating]]
        m = participating[~people.female[participating]]

        # Create preference matrix between eligible females and males that combines age and geo mixing
        age_bins_f = np.digitize(people.age[f],
                                 bins=self.agebins) - 1  # Age bins of females that are entering new relationships
        age_bins_m = np.digitize(people.age[m], bins=self.agebins) - 1  # Age bins of active and participating males
        age_f, age_m = np.meshgrid(age_bins_f, age_bins_m)
        pair_probs = self.pars['mixing'][age_m, age_f + 1]

        f_to_remove = pair_probs.max(axis=0) == 0  # list of female inds to remove if no male partners are found for her
        #f = [i for i, flag in zip(f, f_to_remove) if ~flag]  # remove the inds who don't get paired on this timestep
        f = f[~f_to_remove]
        selected_males = []
        if len(f):
            pair_probs = pair_probs[:, np.invert(f_to_remove)]
            choices = []
            fems = np.arange(len(f))
            f_paired_bools = np.full(len(fems), True, dtype=bool)
            np.random.shuffle(fems) # TODO: Stream-ify?
            for fem in fems:
                m_col = pair_probs[:, fem]
                if m_col.sum() > 0:
                    m_col_norm = m_col / m_col.sum()
                    choice = np.random.choice(len(m_col_norm), p=m_col_norm) # TODO: Stream-ify?
                    choices.append(choice)
                    pair_probs[choice, :] = 0  # Once male partner is assigned, remove from eligible pool
                else:
                    f_paired_bools[fem] = False
            selected_males = m[np.array(choices).flatten()]
            f = f[f_paired_bools]

        p1 = f
        p2 = selected_males
        n_partnerships = len(p1)
        dur = self.pars.duration_dist.rvs(n_partnerships)
        acts = self.pars.act_dist.rvs(n_partnerships)
        age_p1 = people.age[p1]
        age_p2 = people.age[p2]

        age_debut_p1 = people.debut[p1]
        age_debut_p2 = people.debut[p2]

        # For each couple, get the average age they are now and the average age of debut
        avg_age = np.array([age_p1, age_p2]).mean(axis=0)
        avg_debut = np.array([age_debut_p1, age_debut_p2]).mean(axis=0)

        # Shorten parameter names
        dr = self.pars['age_act_pars']['debut_ratio']
        peak = self.pars['age_act_pars']['peak']
        rr = self.pars['age_act_pars']['retirement_ratio']
        retire = self.pars['age_act_pars']['retirement']

        # Get indices of people at different stages
        below_peak_inds = avg_age <= self.pars['age_act_pars']['peak']
        above_peak_inds = (avg_age > self.pars['age_act_pars']['peak']) & (
                avg_age < self.pars['age_act_pars']['retirement'])
        retired_inds = avg_age > self.pars['age_act_pars']['retirement']

        # Set values by linearly scaling the number of acts for each partnership according to
        # the age of the couple at the commencement of the relationship
        below_peak_vals = acts[below_peak_inds] * (dr + (1 - dr) / (peak - avg_debut[below_peak_inds]) * (
                avg_age[below_peak_inds] - avg_debut[below_peak_inds]))
        above_peak_vals = acts[above_peak_inds] * (
                rr + (1 - rr) / (peak - retire) * (avg_age[above_peak_inds] - retire))
        retired_vals = 0

        # Set values and return
        scaled_acts = np.full(len(acts), np.nan, dtype=ss.float_)
        scaled_acts[below_peak_inds] = below_peak_vals
        scaled_acts[above_peak_inds] = above_peak_vals
        scaled_acts[retired_inds] = retired_vals
        start = np.array([ti] * n_partnerships, dtype=ss.float_)
        beta = np.ones(n_partnerships)

        new_contacts = dict(
            p1=p1,
            p2=p2,
            dur=dur,
            acts=scaled_acts,
            start=start,
            beta=beta
        )
        self.append(new_contacts)
        return len(new_contacts)

    def update(self, people, ti=None, dt=None):
        if ti is None: ti = people.ti
        if dt is None: dt = people.dt
        # First remove any relationships due to end
        self.contacts.dur = self.contacts.dur - dt
        active = self.contacts.dur > 0
        for key in self.meta.keys():
            self.contacts[key] = self.contacts[key][active]

        # Then add new relationships
        self.add_pairs(people, ti=ti)
        return


class maternal(Network):
    def __init__(self, key_dict=None, vertical=True, **kwargs):
        """
        Initialized empty and filled with pregnancies throughout the simulation
        """
        key_dict = sc.mergedicts({'dur': ss.float_}, key_dict)
        super().__init__(key_dict=key_dict, vertical=vertical, **kwargs)
        return

    def update(self, people, dt=None):
        if dt is None: dt = people.dt
        # Set beta to 0 for women who complete post-partum period
        # Keep connections for now, might want to consider removing
        self.contacts.dur = self.contacts.dur - dt
        inactive = self.contacts.dur <= 0
        self.contacts.beta[inactive] = 0
        return

    def initialize(self, sim):
        """ No pairs added upon initialization """
        pass

    def add_pairs(self, mother_inds, unborn_inds, dur):
        """
        Add connections between pregnant women and their as-yet-unborn babies
        """
        beta = np.ones_like(mother_inds)
        self.contacts.p1 = np.concatenate([self.contacts.p1, mother_inds])
        self.contacts.p2 = np.concatenate([self.contacts.p2, unborn_inds])
        self.contacts.beta = np.concatenate([self.contacts.beta, beta])
        self.contacts.dur = np.concatenate([self.contacts.dur, dur])
        return len(mother_inds)

class static(Network):
    """
    A network class of static partnerships converted from a networkx graph. There's no formation of new partnerships
    and initialized partnerships only end when one of the partners dies. The networkx graph can be created outside Starsim
    if population size is known. Or the graph can be created by passing a networkx generator function to Starsim.

    **Examples**::

    # Generate a networkx graph and pass to Starsim
    import networkx as nx
    import starsim as ss
    g = nx.scale_free_graph(n=10000)
    ss.static(graph=g)

    # Pass a networkx graph generator to Starsim
    ss.static(graph=nx.erdos_renyi_graph, p=0.0001)

    """
    def __init__(self, graph, **kwargs):
        self.graph = graph
        self.kwargs = kwargs
        super().__init__()
        return

    def initialize(self, sim):
        popsize = sim.pars['n_agents']
        if callable(self.graph):
            self.graph = self.graph(n = popsize, **self.kwargs)
        self.validate_pop(popsize)
        super().initialize(sim)
        self.get_contacts()
        return

    def validate_pop(self, popsize):
        n_nodes =  self.graph.number_of_nodes()
        if n_nodes > popsize:
            errormsg = f'Please ensure the number of nodes in graph {n_nodes} is smaller than population size {popsize}.'
            raise ValueError(errormsg)

    def get_contacts(self):
        p1s = []
        p2s = []
        for edge in self.graph.edges():
            p1, p2 = edge
            p1s.append(p1)
            p2s.append(p2)
        self.contacts.p1 = np.concatenate([self.contacts.p1, p1s])
        self.contacts.p2 = np.concatenate([self.contacts.p2, p2s])
        self.contacts.beta = np.concatenate([self.contacts.beta, np.ones_like(p1s)])
        return


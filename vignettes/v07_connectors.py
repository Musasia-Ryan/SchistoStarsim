"""
Vignette 07: Custom connectors

DECISIONS:
    - all examples follow similar syntax for specifying more than one disease
TO RESOLVE:
    - what are the default connectors?
"""
import stisim as ss

##################
# @robynstuart
##################

# Case 1: modifying simple connectors
syph_pars = dict(init_prev=0.02)
hiv_pars = dict(
    init_prev=0.02,
    # Here come the connectors - simple connectors are multipliers on any people attribute (e.g. people.hiv.rel_trans)
    #   - syphilis increases risk of acquiring HIV x2
    #   - syphilis increases risk of transmitting HIV x3
    #   - syphilis increases HIV viral load x10
    syphilis={'rel_trans': 2, 'rel_sus': 3, 'viral_load': 10},
)
syph = ss.syphilis(pars=syph_pars)
hiv = ss.syphilis(pars=hiv_pars)
sim = ss.Sim(modules=[syph, hiv])
sim.initialize()  # Adds modules and parameters
# Parameters can be read like a conditional probability, i.e. prob(transmitting hiv | syphilis infection) = 2
sim['hiv']['syphilis']['rel_trans'] = 4

# Case 2: creating complex connectors
# Example where rel_trans is higher for those with lower cd4 count...
complex_connector = lambda rel_trans, cd4: (6*rel_trans**2 * (cd4<500)) + (2*rel_trans * (cd4>500))
hiv_pars = dict(
    init_prev=0.02,
    syphilis={'rel_trans': complex_connector},
)
syph = ss.syphilis(pars=syph_pars)
hiv = ss.syphilis(pars=hiv_pars)
sim = ss.Sim(modules=[syph, hiv])

# Case 3: creating even more complex connectors
# Completely made-up example where HIV makes syphilis incurable and permanently latent.
# At this stage, I feel that most examples of connectors could be managed by modifying the people
# attributes, so this would be a relatively rare example. If it became more common we could include
# defaults. I think that the intervention class would be able to handle this as-is, but for clarity
# we could make a Connector subclass.
class hiv_syph_effect(ss.Intervention):
    def __init__(self):
        pass
    def initialize(self, sim):
        pass
    def apply(self, sim):
        hiv_syph_inds = sim.people.hiv.infected * sim.people.syphilis.infected
        sim.people.syphilis.dur_latent[hiv_syph_inds] = 100
        sim.people.syphilis.treatable[hiv_syph_inds] = False


# Case 4: creating even MORE complex connectors
# For anything more complex than this, modify the module directly, i.e.
class complex_hiv_syphilis(ss.HIV):
    # Write your own!
    pass


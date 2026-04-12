import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def get_fuzzy_risk(toxicity_score, sentiment_score):
    # Define Universe
    tox = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'toxicity')
    sent = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'sentiment')
    risk = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'risk')

    # Gaussian Membership Functions
    tox['low'] = fuzz.gaussmf(tox.universe, 0, 0.2)
    tox['med'] = fuzz.gaussmf(tox.universe, 0.5, 0.2)
    tox['high'] = fuzz.gaussmf(tox.universe, 1, 0.2)

    sent['pos'] = fuzz.gaussmf(sent.universe, 0, 0.2)
    sent['neu'] = fuzz.gaussmf(sent.universe, 0.5, 0.2)
    sent['neg'] = fuzz.gaussmf(sent.universe, 1, 0.2)

    risk['low'] = fuzz.trimf(risk.universe, [0, 0, 0.5])
    risk['med'] = fuzz.trimf(risk.universe, [0, 0.5, 1])
    risk['high'] = fuzz.trimf(risk.universe, [0.5, 1, 1])

    # Rules
    rule1 = ctrl.Rule(tox['high'] | sent['neg'], risk['high'])
    rule2 = ctrl.Rule(tox['med'] & sent['neu'], risk['med'])
    rule3 = ctrl.Rule(tox['low'] & sent['pos'], risk['low'])

    risk_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
    sim = ctrl.ControlSystemSimulation(risk_ctrl)
    
    sim.input['toxicity'] = toxicity_score
    sim.input['sentiment'] = sentiment_score
    sim.compute()
    
    return sim.output['risk'], toxicity_score, sentiment_score
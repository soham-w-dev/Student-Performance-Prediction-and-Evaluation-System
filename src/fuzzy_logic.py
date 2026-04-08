import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def get_fuzzy_recommendation(study_hours_val, sleep_hours_val, prev_scores_val):
    # 1. Define Antecedents (Inputs) and Consequents (Outputs)
    study = ctrl.Antecedent(np.arange(0, 25, 1), 'study')
    sleep = ctrl.Antecedent(np.arange(0, 13, 1), 'sleep')
    scores = ctrl.Antecedent(np.arange(0, 101, 1), 'scores')
    
    recommendation_score = ctrl.Consequent(np.arange(0, 101, 1), 'recommendation_score')

    # 2. Define Custom Membership Functions for Inputs (instead of automf which divides blindly)
    # Most students study 3-8 hours. 24 hours is impossible. We adjust the math to real life.
    study['low'] = fuzz.trimf(study.universe, [0, 0, 5])
    study['medium'] = fuzz.trimf(study.universe, [3, 6, 9])
    study['high'] = fuzz.trimf(study.universe, [7, 12, 24]) # 7+ hours is considered high

    sleep['deficient'] = fuzz.trimf(sleep.universe, [0, 0, 5]) # 5 hours is no longer 'deficient'
    sleep['normal'] = fuzz.trimf(sleep.universe, [4, 7, 10]) # 4 to 10 hours covers typical student normal
    sleep['excessive'] = fuzz.trimf(sleep.universe, [9, 12, 12])

    scores['poor'] = fuzz.trimf(scores.universe, [0, 0, 55])
    scores['average'] = fuzz.trimf(scores.universe, [50, 70, 85])
    scores['excellent'] = fuzz.trimf(scores.universe, [80, 100, 100])
    
    # 3. Define Custom Membership Functions for Output
    recommendation_score['bad'] = fuzz.trimf(recommendation_score.universe, [0, 0, 50])
    recommendation_score['improve'] = fuzz.trimf(recommendation_score.universe, [0, 50, 100])
    recommendation_score['great'] = fuzz.trimf(recommendation_score.universe, [50, 100, 100])

    # 4. Define Fuzzy Rules (spelling out ORs to avoid scikit-fuzzy bitwise bugs)
    rule1 = ctrl.Rule(study['low'] & scores['poor'], recommendation_score['bad'])
    
    rule2 = ctrl.Rule(study['medium'] & sleep['normal'] & scores['average'], recommendation_score['improve'])
    rule3 = ctrl.Rule(study['medium'] & sleep['normal'] & scores['excellent'], recommendation_score['improve'])
    
    rule4 = ctrl.Rule(study['high'] & scores['average'], recommendation_score['great'])
    rule5 = ctrl.Rule(study['high'] & scores['excellent'], recommendation_score['great'])
    
    rule6 = ctrl.Rule(study['high'] & sleep['deficient'], recommendation_score['improve'])
    rule7 = ctrl.Rule(sleep['excessive'], recommendation_score['improve'])
    
    # 5. Create Control System
    recommender_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7])
    recommender_sim = ctrl.ControlSystemSimulation(recommender_ctrl)
    
    # 6. Feed Inputs and Compute
    recommender_sim.input['study'] = study_hours_val
    recommender_sim.input['sleep'] = sleep_hours_val
    recommender_sim.input['scores'] = prev_scores_val
    
    try:
        recommender_sim.compute()
        output_score = recommender_sim.output['recommendation_score']
    except (ValueError, KeyError, Exception):
        # Fallback if inputs fall completely into a "gap" between rules
        output_score = 60.0 
    
    # 7. Provide Semantic Text Translation
    if output_score < 40:
        text = "⚠️ Needs Action: Your current routine is suboptimal. Check your sleep and study balance."
    elif output_score < 75:
        text = "👍 Doing Okay: But there is room for improvement in your study-sleep balance to maximize scores."
    else:
        text = "🌟 Excellent Routine! You have a great balance. Keep up the good work."
        
    return output_score, text

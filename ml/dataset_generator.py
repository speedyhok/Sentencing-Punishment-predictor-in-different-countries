# ==============================================================================
# Global Sentencing Proportionality Drift Auditor
# Copyright (c) 2026 Mohibul Hoque
# Licensed under the MIT License (see LICENSE file for details)
# Author: Mohibul Hoque <hokworks@gmail.com> (github.com/speedyhok | linkedin.com/in/speedymohibul)
# Description: Modular machine learning pipeline for cross-jurisdiction legal auditing.
# ==============================================================================

import os
import random
import csv
import numpy as np

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Define the 20 countries, their legal systems, and their parameters
# Weights for: [violence, financial_loss, vulnerability, premeditation, public_safety_risk]
# Multipliers/discounts: [priors_coefficient, plea_discount, mitigation_discount, juvenile_discount]
# regional_bias: coefficient representing regional severity shift (0 = Rural, 1 = Metro)
# noise_sigma: standard deviation in log-space representing proportional variance
# max_sentence: maximum capped sentence in months
COUNTRY_PROFILES = {
    "United States": {
        "legal_system": "Common Law",
        "weights": [120.0, 60.0, 30.0, 40.0, 50.0],
        "priors_coeff": 0.5,
        "plea_discount": 0.70, # 30% reduction
        "mitigation_discount": 0.90,
        "juvenile_discount": 0.50,
        "regional_bias": -0.15, # Metro is 15% more lenient
        "noise_sigma": 0.15,
        "max_sentence": 360.0,
        "description": "Steep sentencing guidelines with heavy penalties for prior offenses and significant discounts for plea bargaining."
    },
    "United Kingdom": {
        "legal_system": "Common Law",
        "weights": [90.0, 40.0, 24.0, 30.0, 36.0],
        "priors_coeff": 0.3,
        "plea_discount": 0.67, # 33% reduction (standard early guilty plea)
        "mitigation_discount": 0.80,
        "juvenile_discount": 0.40,
        "regional_bias": -0.08, # Metro is 8% more lenient
        "noise_sigma": 0.12,
        "max_sentence": 240.0,
        "description": "Structured sentencing guidelines based on harm and culpability, offering formulaic discounts for early pleas."
    },
    "Canada": {
        "legal_system": "Common Law",
        "weights": [70.0, 30.0, 20.0, 25.0, 24.0],
        "priors_coeff": 0.25,
        "plea_discount": 0.75,
        "mitigation_discount": 0.75,
        "juvenile_discount": 0.40,
        "regional_bias": -0.06,
        "noise_sigma": 0.10,
        "max_sentence": 240.0,
        "description": "Proportional sentencing framework with strong emphasis on indigenous background considerations and rehabilitation."
    },
    "Australia": {
        "legal_system": "Common Law",
        "weights": [75.0, 35.0, 22.0, 25.0, 28.0],
        "priors_coeff": 0.28,
        "plea_discount": 0.72,
        "mitigation_discount": 0.78,
        "juvenile_discount": 0.40,
        "regional_bias": -0.07,
        "noise_sigma": 0.11,
        "max_sentence": 240.0,
        "description": "Combines statutory guidance and common law discretion, focusing heavily on community protection and proportional deterrence."
    },
    "Singapore": {
        "legal_system": "Common Law / Mixed",
        "weights": [150.0, 50.0, 40.0, 50.0, 120.0],
        "priors_coeff": 0.4,
        "plea_discount": 0.85,
        "mitigation_discount": 0.95,
        "juvenile_discount": 0.60,
        "regional_bias": 0.0, # City-state: no regional bias!
        "noise_sigma": 0.08,
        "max_sentence": 480.0,
        "description": "Highly punitive system emphasizing strict deterrence, especially for drug-related offenses and weapons."
    },
    "India": {
        "legal_system": "Common Law / Mixed",
        "weights": [80.0, 40.0, 25.0, 30.0, 35.0],
        "priors_coeff": 0.25,
        "plea_discount": 0.80,
        "mitigation_discount": 0.75,
        "juvenile_discount": 0.40,
        "regional_bias": 0.20, # Higher sentences in metropolitan centers due to specific crime control acts
        "noise_sigma": 0.22, # High log-normal variance
        "max_sentence": 240.0,
        "description": "High judicial discretion with significant sentencing variance, reflecting systemic complexities and backlog."
    },
    "Germany": {
        "legal_system": "Civil Law",
        "weights": [60.0, 30.0, 20.0, 25.0, 24.0],
        "priors_coeff": 0.25,
        "plea_discount": 0.80,
        "mitigation_discount": 0.70,
        "juvenile_discount": 0.40,
        "regional_bias": -0.05,
        "noise_sigma": 0.08,
        "max_sentence": 180.0,
        "description": "Codified civil law focusing on proportional guilt and rehabilitation. Relies heavily on mitigating circumstances."
    },
    "France": {
        "legal_system": "Civil Law",
        "weights": [54.0, 24.0, 20.0, 20.0, 24.0],
        "priors_coeff": 0.30,
        "plea_discount": 0.85,
        "mitigation_discount": 0.70,
        "juvenile_discount": 0.50,
        "regional_bias": -0.04,
        "noise_sigma": 0.08,
        "max_sentence": 180.0,
        "description": "Structured code where repeat offenses (récidive) trigger statutory minimums, but judges retain mitigation flexibility."
    },
    "Italy": {
        "legal_system": "Civil Law",
        "weights": [58.0, 28.0, 18.0, 22.0, 22.0],
        "priors_coeff": 0.22,
        "plea_discount": 0.67, # patteggiamento yields exactly 1/3 reduction
        "mitigation_discount": 0.75,
        "juvenile_discount": 0.45,
        "regional_bias": -0.06,
        "noise_sigma": 0.09,
        "max_sentence": 240.0,
        "description": "Highly codified civil system with structured sentence reductions, notably for plea bargains (patteggiamento)."
    },
    "Spain": {
        "legal_system": "Civil Law",
        "weights": [56.0, 26.0, 18.0, 22.0, 22.0],
        "priors_coeff": 0.24,
        "plea_discount": 0.75,
        "mitigation_discount": 0.72,
        "juvenile_discount": 0.42,
        "regional_bias": -0.05,
        "noise_sigma": 0.08,
        "max_sentence": 240.0,
        "description": "Strict legal codes with specific thresholds for penalties and moderately high emphasis on restorative elements."
    },
    "Switzerland": {
        "legal_system": "Civil Law",
        "weights": [45.0, 20.0, 15.0, 15.0, 15.0],
        "priors_coeff": 0.20,
        "plea_discount": 0.80,
        "mitigation_discount": 0.65,
        "juvenile_discount": 0.35,
        "regional_bias": -0.03,
        "noise_sigma": 0.06, # Consistent
        "max_sentence": 240.0,
        "description": "Exceptionally consistent civil law country with low overall sentencing terms and high focus on rehabilitation."
    },
    "Brazil": {
        "legal_system": "Civil Law / Latin American",
        "weights": [65.0, 30.0, 20.0, 20.0, 24.0],
        "priors_coeff": 0.22,
        "plea_discount": 0.80,
        "mitigation_discount": 0.75,
        "juvenile_discount": 0.45,
        "regional_bias": 0.10, # Metropolitan hubs are slightly more severe
        "noise_sigma": 0.12,
        "max_sentence": 360.0,
        "description": "Influenced by Latin American civil law; sentencing ranges are broad with statutory maximums capped at 30 years."
    },
    "Norway": {
        "legal_system": "Nordic Civil Law",
        "weights": [36.0, 12.0, 10.0, 8.0, 12.0],
        "priors_coeff": 0.15,
        "plea_discount": 0.85,
        "mitigation_discount": 0.60,
        "juvenile_discount": 0.30,
        "regional_bias": -0.02, # Very low regional variation
        "noise_sigma": 0.05, # Extremely consistent
        "max_sentence": 252.0,
        "description": "Pioneering rehabilitative system with extremely flat sentencing response curves and low overall prison terms."
    },
    "Sweden": {
        "legal_system": "Nordic Civil Law",
        "weights": [38.0, 14.0, 10.0, 8.0, 12.0],
        "priors_coeff": 0.18,
        "plea_discount": 0.85,
        "mitigation_discount": 0.65,
        "juvenile_discount": 0.30,
        "regional_bias": -0.03,
        "noise_sigma": 0.06,
        "max_sentence": 240.0,
        "description": "Nordic model focused on rehabilitation, utilizing a structured scale of 'penal value' for offenses."
    },
    "Denmark": {
        "legal_system": "Nordic Civil Law",
        "weights": [40.0, 12.0, 10.0, 8.0, 12.0],
        "priors_coeff": 0.16,
        "plea_discount": 0.85,
        "mitigation_discount": 0.65,
        "juvenile_discount": 0.35,
        "regional_bias": -0.02,
        "noise_sigma": 0.06,
        "max_sentence": 240.0,
        "description": "Nordic model balancing social reintegration with structured penalty guidelines."
    },
    "Japan": {
        "legal_system": "Civil Law / East Asian",
        "weights": [48.0, 24.0, 18.0, 20.0, 20.0],
        "priors_coeff": 0.20,
        "plea_discount": 0.50, # apology and settlement discount (50% reduction)
        "mitigation_discount": 0.60,
        "juvenile_discount": 0.40,
        "regional_bias": -0.06,
        "noise_sigma": 0.07,
        "max_sentence": 240.0,
        "description": "Precision justice system where confession, sincere remorse, and victim restitution lead to dramatic sentence discounts."
    },
    "South Korea": {
        "legal_system": "Civil Law / East Asian",
        "weights": [52.0, 25.0, 20.0, 22.0, 22.0],
        "priors_coeff": 0.22,
        "plea_discount": 0.70,
        "mitigation_discount": 0.65,
        "juvenile_discount": 0.40,
        "regional_bias": -0.05,
        "noise_sigma": 0.08,
        "max_sentence": 240.0,
        "description": "Influenced by German civil law with structured sentencing guidelines. Remorse and deposit of settlement money are key."
    },
    "Saudi Arabia": {
        "legal_system": "Islamic Law",
        "weights": [140.0, 80.0, 30.0, 40.0, 60.0],
        "priors_coeff": 0.3,
        "plea_discount": 0.90,
        "mitigation_discount": 0.90,
        "juvenile_discount": 0.70,
        "regional_bias": -0.10,
        "noise_sigma": 0.15,
        "max_sentence": 360.0,
        "description": "Based on Sharia legal principles; emphasizes heavy retributive punishment (Qisas/Tazir) for violent and financial crimes."
    },
    "Iran": {
        "legal_system": "Islamic Law",
        "weights": [130.0, 70.0, 25.0, 35.0, 50.0],
        "priors_coeff": 0.3,
        "plea_discount": 0.90,
        "mitigation_discount": 0.95,
        "juvenile_discount": 0.75,
        "regional_bias": -0.08,
        "noise_sigma": 0.14,
        "max_sentence": 360.0,
        "description": "Codified Islamic penal system with high severity weights and conservative discounts for mitigating factors."
    },
    "South Africa": {
        "legal_system": "Mixed (Civil/Common/Customary)",
        "weights": [95.0, 45.0, 25.0, 35.0, 45.0],
        "priors_coeff": 0.32,
        "plea_discount": 0.80,
        "mitigation_discount": 0.80,
        "juvenile_discount": 0.50,
        "regional_bias": -0.05,
        "noise_sigma": 0.13,
        "max_sentence": 300.0,
        "description": "Combines Roman-Dutch civil law and English common law, featuring statutory minimum sentences for severe crimes."
    }
}

# Template components for synthetic text generation (same as before)
CRIME_TEMPLATES = {
    "violence": [
        {"text": "The offender physically assaulted the victim by punching and kicking them multiple times, causing severe facial fractures.",
         "scores": {"violence": 0.7, "financial": 0.0, "vulnerability": 0.1, "premeditation": 0.2, "safety": 0.4}},
        {"text": "The defendant attacked the victim with a metal pipe during an argument, resulting in a deep head wound.",
         "scores": {"violence": 0.8, "financial": 0.0, "vulnerability": 0.1, "premeditation": 0.3, "safety": 0.5}},
        {"text": "The suspect pointed a loaded handgun at the store clerk, demanding money and threatening to pull the trigger.",
         "scores": {"violence": 0.85, "financial": 0.2, "vulnerability": 0.2, "premeditation": 0.6, "safety": 0.7}},
        {"text": "A verbal confrontation escalated, leading to the offender shoving the victim to the ground, causing minor bruises.",
         "scores": {"violence": 0.3, "financial": 0.0, "vulnerability": 0.0, "premeditation": 0.0, "safety": 0.1}},
        {"text": "The perpetrator shot the victim in the chest at close range following a dispute, causing fatal injuries.",
         "scores": {"violence": 1.0, "financial": 0.0, "vulnerability": 0.2, "premeditation": 0.5, "safety": 0.9}}
    ],
    "financial": [
        {"text": "The employee embezzled $150,000 from the company accounts by forging signatures on corporate checks over two years.",
         "scores": {"violence": 0.0, "financial": 0.75, "vulnerability": 0.3, "premeditation": 0.8, "safety": 0.2}},
        {"text": "The suspect broke into a commercial warehouse and stole electronics worth approximately $5,000.",
         "scores": {"violence": 0.1, "financial": 0.35, "vulnerability": 0.1, "premeditation": 0.5, "safety": 0.2}},
        {"text": "The offender stole a bicycle valued at $300 parked outside a supermarket.",
         "scores": {"violence": 0.0, "financial": 0.08, "vulnerability": 0.0, "premeditation": 0.1, "safety": 0.05}},
        {"text": "The defendant ran a sophisticated Ponzi scheme, defrauding elderly investors of over $2.5 million.",
         "scores": {"violence": 0.0, "financial": 0.95, "vulnerability": 0.8, "premeditation": 0.9, "safety": 0.4}},
        {"text": "The suspect stole a wallet containing credit cards and $150 cash from an unattended purse in a cafe.",
         "scores": {"violence": 0.0, "financial": 0.09, "vulnerability": 0.0, "premeditation": 0.2, "safety": 0.05}}
    ],
    "vulnerability": [
        {"text": "The victim was an 82-year-old grandmother who lived alone and suffered from mild dementia.",
         "scores": {"violence": 0.0, "financial": 0.0, "vulnerability": 0.9, "premeditation": 0.2, "safety": 0.1}},
        {"text": "The crime targeted a 7-year-old child who was walking home alone from school.",
         "scores": {"violence": 0.1, "financial": 0.0, "vulnerability": 0.95, "premeditation": 0.3, "safety": 0.3}},
        {"text": "The defendant took advantage of their role as a live-in nurse to abuse and steal from a bedridden patient.",
         "scores": {"violence": 0.2, "financial": 0.2, "vulnerability": 0.9, "premeditation": 0.6, "safety": 0.2}},
        {"text": "The offender chose a victim who was highly intoxicated and unable to stand or speak clearly.",
         "scores": {"violence": 0.1, "financial": 0.1, "vulnerability": 0.7, "premeditation": 0.2, "safety": 0.1}}
    ],
    "premeditation": [
        {"text": "The suspect planned the heist for over six months, purchasing specialized tools, drawing maps, and tracking guards.",
         "scores": {"violence": 0.0, "financial": 0.3, "vulnerability": 0.1, "premeditation": 0.95, "safety": 0.4}},
        {"text": "The offender acted in a sudden fit of rage, with no planning whatsoever, responding to an insult.",
         "scores": {"violence": 0.4, "financial": 0.0, "vulnerability": 0.0, "premeditation": 0.0, "safety": 0.1}},
        {"text": "The defendant purchased a carving knife and gloves the morning of the crime, explicitly intending to confront the victim.",
         "scores": {"violence": 0.6, "financial": 0.0, "vulnerability": 0.2, "premeditation": 0.85, "safety": 0.4}},
        {"text": "The perpetrator wore a ski mask and dark clothing to conceal their identity and avoid CCTV detection.",
         "scores": {"violence": 0.1, "financial": 0.1, "vulnerability": 0.1, "premeditation": 0.7, "safety": 0.2}}
    ],
    "safety": [
        {"text": "The suspect set fire to a trash can near an apartment building, causing a blaze that threatened multiple families.",
         "scores": {"violence": 0.4, "financial": 0.3, "vulnerability": 0.5, "premeditation": 0.4, "safety": 0.85}},
        {"text": "The offender was found in possession of 200 grams of cocaine, scales, and packaging materials indicating distribution.",
         "scores": {"violence": 0.1, "financial": 0.2, "vulnerability": 0.1, "premeditation": 0.5, "safety": 0.75}},
        {"text": "The defendant discharged an assault rifle into the air in a crowded public square, causing mass panic.",
         "scores": {"violence": 0.5, "financial": 0.0, "vulnerability": 0.3, "premeditation": 0.4, "safety": 0.95}},
        {"text": "The suspect sold fake pharmaceutical pills containing toxic contaminants to local drug users.",
         "scores": {"violence": 0.3, "financial": 0.4, "vulnerability": 0.7, "premeditation": 0.7, "safety": 0.9}}
    ]
}

# Helper to generate case sentences using log-normal noise and region features
def generate_case(case_id, perturb=False):
    mix_type = random.choice(["violence_only", "financial_only", "violent_robbery", "complex_white_collar", "drug_offense", "minor_crime", "general"])
    selected_components = []
    
    if mix_type == "violence_only":
        selected_components.append(random.choice(CRIME_TEMPLATES["violence"]))
        if random.random() < 0.6:
            selected_components.append(random.choice(CRIME_TEMPLATES["premeditation"]))
        if random.random() < 0.4:
            selected_components.append(random.choice(CRIME_TEMPLATES["vulnerability"]))
    elif mix_type == "financial_only":
        selected_components.append(random.choice(CRIME_TEMPLATES["financial"]))
        if random.random() < 0.7:
            selected_components.append(random.choice(CRIME_TEMPLATES["premeditation"]))
    elif mix_type == "violent_robbery":
        selected_components.append(random.choice(CRIME_TEMPLATES["violence"]))
        selected_components.append(random.choice(CRIME_TEMPLATES["financial"]))
        if random.random() < 0.5:
            selected_components.append(random.choice(CRIME_TEMPLATES["vulnerability"]))
    elif mix_type == "complex_white_collar":
        selected_components.append(random.choice(CRIME_TEMPLATES["financial"]))
        selected_components.append(random.choice(CRIME_TEMPLATES["premeditation"]))
        selected_components.append(random.choice(CRIME_TEMPLATES["vulnerability"]))
    elif mix_type == "drug_offense":
        selected_components.append(random.choice(CRIME_TEMPLATES["safety"]))
        if random.random() < 0.5:
            selected_components.append(random.choice(CRIME_TEMPLATES["premeditation"]))
    elif mix_type == "minor_crime":
        minor_v = [t for t in CRIME_TEMPLATES["violence"] if t["scores"]["violence"] < 0.5]
        minor_f = [t for t in CRIME_TEMPLATES["financial"] if t["scores"]["financial"] < 0.3]
        selected_components.append(random.choice(minor_v if random.random() < 0.5 else minor_f))
    else: # general
        keys = list(CRIME_TEMPLATES.keys())
        chosen_keys = random.sample(keys, random.randint(2, 3))
        for k in chosen_keys:
            selected_components.append(random.choice(CRIME_TEMPLATES[k]))
            
    # Combine text
    texts = [comp["text"] for comp in selected_components]
    fact_pattern = " ".join(texts)
    
    # Calculate severity scores
    scores = {"violence": 0.0, "financial": 0.0, "vulnerability": 0.0, "premeditation": 0.0, "safety": 0.0}
    for comp in selected_components:
        for k in scores.keys():
            scores[k] = max(scores[k], comp["scores"][k])
            
    for k in scores.keys():
        scores[k] = min(1.0, max(0.0, scores[k] + random.uniform(-0.05, 0.05)))
        
    priors = random.choices([0, 1, 2, 3, 4, 5], weights=[0.45, 0.25, 0.15, 0.08, 0.05, 0.02])[0]
    plea_guilty = random.choices([0, 1], weights=[0.25, 0.75])[0]
    mitigating_circumstances = random.choices([0, 1], weights=[0.6, 0.4])[0]
    juvenile = random.choices([0, 1], weights=[0.95, 0.05])[0]
    court_region = random.choice([0, 1]) # 0 = Rural, 1 = Metropolitan
    
    case = {
        "case_id": case_id,
        "fact_pattern": fact_pattern,
        "violence_score": round(scores["violence"], 3),
        "financial_loss_score": round(scores["financial"], 3),
        "victim_vulnerability_score": round(scores["vulnerability"], 3),
        "premeditation_score": round(scores["premeditation"], 3),
        "public_safety_risk": round(scores["safety"], 3),
        "priors": priors,
        "plea_guilty": plea_guilty,
        "mitigating_circumstances": mitigating_circumstances,
        "juvenile": juvenile,
        "court_region": court_region
    }
    
    # Calculate sentences
    for country, profile in COUNTRY_PROFILES.items():
        w = np.array(profile["weights"])
        priors_coeff = profile["priors_coeff"]
        plea_discount = profile["plea_discount"]
        mitigation_discount = profile["mitigation_discount"]
        juvenile_discount = profile["juvenile_discount"]
        regional_bias = profile["regional_bias"]
        noise_sigma = profile["noise_sigma"]
        max_sentence = profile["max_sentence"]
        
        if perturb:
            # Shift coefficients randomly by up to 15% for the leakage test
            w = w * np.random.uniform(0.85, 1.15, size=5)
            priors_coeff *= np.random.uniform(0.85, 1.15)
            plea_discount = min(1.0, plea_discount * np.random.uniform(0.85, 1.15))
            mitigation_discount = min(1.0, mitigation_discount * np.random.uniform(0.85, 1.15))
            regional_bias *= np.random.uniform(0.85, 1.15)
            noise_sigma *= np.random.uniform(0.85, 1.15)
            
        base_sentence = (
            case["violence_score"] * w[0] +
            case["financial_loss_score"] * w[1] +
            case["victim_vulnerability_score"] * w[2] +
            case["premeditation_score"] * w[3] +
            case["public_safety_risk"] * w[4]
        )
        
        # Apply legal factors
        factor = 1.0 + (priors * priors_coeff)
        if plea_guilty == 1:
            factor *= plea_discount
        if mitigating_circumstances == 1:
            factor *= mitigation_discount
        if juvenile == 1:
            factor *= juvenile_discount
            
        # Apply Regional Bias
        factor *= (1.0 + court_region * regional_bias)
        
        sentence = base_sentence * factor
        
        # Apply Multiplicative Log-Normal Noise
        # sentence = sentence * exp(epsilon), epsilon ~ N(0, noise_sigma)
        noise_factor = np.random.lognormal(mean=0.0, sigma=noise_sigma)
        sentence = sentence * noise_factor
        
        sentence = min(sentence, max_sentence)
        case[f"sentence_{country}"] = round(max(0.0, sentence), 1)
        
    return case

def main():
    print("Generating refined synthetic crime cases dataset (log-normal noise + region)...")
    os.makedirs("ml", exist_ok=True)
    
    num_cases = 1000
    cases = []
    for i in range(num_cases):
        cases.append(generate_case(i, perturb=False))
        
    fieldnames = list(cases[0].keys())
    csv_path = os.path.join("ml", "synthetic_cases.csv")
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cases)
        
    print(f"Refined dataset saved to {csv_path} with {num_cases} entries.")
    
if __name__ == "__main__":
    main()

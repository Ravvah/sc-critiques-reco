#!/usr/bin/env python3
"""
Script de lancement rapide pour l'évaluation.
"""

import sys
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

from evaluation import UnsupervisedRecommendationEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Lancement de l'évaluation du système de recommandation")
    logger.info("Mode évaluation non supervisée (métriques standards)")
    
    evaluator = UnsupervisedRecommendationEvaluator()
    results = evaluator.run_unsupervised_evaluation()
    
    if results:
        logger.info("Évaluation terminée avec succès")
        logger.info("Consultez les résultats détaillés dans unsupervised_evaluation_results.json")
    else:
        logger.error("Évaluation échouée")
        logger.info("Vérifiez que l'API est lancée: uvicorn app.api:app --reload")
        logger.info("Vérifiez que l'endpoint /v1/readyz retourne ready: true")

if __name__ == "__main__":
    main()
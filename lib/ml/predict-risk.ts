import type { IncidentState, MLResult, RiskLevel } from "@/lib/types";
import { DEFAULT_MODEL_VERSION } from "@/lib/constants";
import { approximateFeatureImportance } from "@/lib/ml/feature-importance";
import { incidentToFeatures } from "@/lib/ml/dataset";

export function riskLevelFromProbability(probability: number): RiskLevel {
  if (probability >= 0.82) return "critical";
  if (probability >= 0.62) return "high";
  if (probability >= 0.35) return "medium";
  return "low";
}

export function heuristicRiskPrediction(incident: IncidentState): MLResult {
  const features = incidentToFeatures(incident);
  const score =
    features[0] * 1.25 +
    features[1] * 1.1 +
    features[2] * 0.8 +
    features[3] * 1.15 +
    (1 - features[4]) * 0.25 -
    features[5] * 0.65 +
    features[6] * 0.1 +
    features[7] * 0.08 -
    1.3;
  const probability = 1 / (1 + Math.exp(-score));

  return {
    fireProbability: Number(probability.toFixed(3)),
    riskLevel: riskLevelFromProbability(probability),
    modelVersion: DEFAULT_MODEL_VERSION,
    featureImportance: approximateFeatureImportance(features),
    explanation:
      "The ML model predicts fire probability from fused evidence. It does not execute tools, unlock gates, notify authorities, or dispatch drones."
  };
}

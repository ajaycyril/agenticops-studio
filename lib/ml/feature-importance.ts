import { featureNames } from "@/lib/ml/dataset";

export function approximateFeatureImportance(features: number[]) {
  const weights = [1.25, 1.1, 0.8, 1.15, 0.25, 0.65, 0.1, 0.08];
  const raw = featureNames.map((feature, index) => ({
    feature,
    importance: Math.abs(features[index] * weights[index])
  }));
  const total = raw.reduce((sum, item) => sum + item.importance, 0) || 1;
  return raw
    .map((item) => ({ ...item, importance: Number((item.importance / total).toFixed(3)) }))
    .sort((a, b) => b.importance - a.importance);
}

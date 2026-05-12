export function classificationMetrics(labels: number[], predictions: number[]) {
  let tp = 0;
  let tn = 0;
  let fp = 0;
  let fn = 0;

  labels.forEach((label, index) => {
    const prediction = predictions[index] >= 0.5 ? 1 : 0;
    if (label === 1 && prediction === 1) tp += 1;
    if (label === 0 && prediction === 0) tn += 1;
    if (label === 0 && prediction === 1) fp += 1;
    if (label === 1 && prediction === 0) fn += 1;
  });

  const total = labels.length || 1;
  return {
    accuracy: (tp + tn) / total,
    precision: tp / Math.max(tp + fp, 1),
    recall: tp / Math.max(tp + fn, 1),
    falsePositiveRate: fp / Math.max(fp + tn, 1)
  };
}

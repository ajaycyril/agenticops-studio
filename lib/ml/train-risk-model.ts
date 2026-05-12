"use client";

import * as tf from "@tensorflow/tfjs";
import { generateSyntheticDataset } from "@/lib/ml/dataset";
import { classificationMetrics } from "@/lib/ml/metrics";

export type TrainingProgress = {
  epoch: number;
  loss: number;
};

export async function trainRiskModel(params: {
  size: number;
  epochs: number;
  falseAlarmBias: number;
  onProgress?: (point: TrainingProgress) => void;
}) {
  await tf.ready();
  const data = generateSyntheticDataset(params.size, params.falseAlarmBias);
  const xs = tf.tensor2d(data.map((row) => row.features));
  const ys = tf.tensor2d(data.map((row) => [row.label]));

  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [8], units: 1, activation: "sigmoid" }));
  model.compile({ optimizer: tf.train.adam(0.08), loss: "binaryCrossentropy", metrics: ["accuracy"] });

  await model.fit(xs, ys, {
    epochs: params.epochs,
    shuffle: true,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        params.onProgress?.({ epoch: epoch + 1, loss: Number((logs?.loss ?? 0).toFixed(4)) });
        await tf.nextFrame();
      }
    }
  });

  const predictions = Array.from((model.predict(xs) as tf.Tensor).dataSync());
  const labels = data.map((row) => row.label);
  const metrics = classificationMetrics(labels, predictions);
  xs.dispose();
  ys.dispose();

  return {
    model,
    metrics,
    modelVersion: `tfjs-risk-model.demo.${Date.now()}`
  };
}

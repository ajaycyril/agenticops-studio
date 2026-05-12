import type { IncidentState } from "@/lib/types";

export const featureNames = [
  "smokePpm",
  "temperatureC",
  "cameraSmokeConfidence",
  "cameraFireConfidence",
  "sensorHealth",
  "historicalFalseAlarmRate",
  "occupancyEncoded",
  "windSpeedKmh"
] as const;

export type RiskFeatureName = (typeof featureNames)[number];

export type TrainingExample = {
  features: number[];
  label: 0 | 1;
};

export function occupancyEncoded(status: IncidentState["occupancyStatus"]) {
  if (status === "detected") return 1;
  if (status === "unknown") return 0.5;
  return 0;
}

export function incidentToFeatures(incident: IncidentState): number[] {
  return [
    incident.smokePpm / 140,
    incident.temperatureC / 80,
    incident.cameraSmokeConfidence,
    incident.cameraFireConfidence,
    incident.sensorHealth,
    incident.historicalFalseAlarmRate,
    occupancyEncoded(incident.occupancyStatus),
    incident.windSpeedKmh / 40
  ];
}

export function generateSyntheticDataset(size = 220, falseAlarmBias = 0.25): TrainingExample[] {
  return Array.from({ length: size }, () => {
    const smoke = Math.random();
    const heat = Math.random();
    const camSmoke = Math.random();
    const camFire = Math.random();
    const sensorHealth = 0.45 + Math.random() * 0.55;
    const falseAlarmRate = Math.random();
    const occupancy = Math.random();
    const wind = Math.random();

    const score =
      smoke * 1.25 +
      heat * 1.1 +
      camSmoke * 0.8 +
      camFire * 1.15 +
      (1 - sensorHealth) * 0.25 -
      falseAlarmRate * (0.65 + falseAlarmBias) +
      occupancy * 0.1 +
      wind * 0.08 +
      (Math.random() - 0.5) * 0.4;

    return {
      features: [smoke, heat, camSmoke, camFire, sensorHealth, falseAlarmRate, occupancy, wind],
      label: score > 2.05 ? 1 : 0
    };
  });
}

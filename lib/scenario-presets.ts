import type { IncidentState } from "@/lib/types";

const base = {
  siteId: "villa-cluster-07",
  droneAvailable: true,
  gateLocked: true,
  imageUrl: "/sample-images/unclear-camera.svg",
  visionProvider: "sample" as const,
  visionDetections: [],
  humanApproval: {
    unlockGate: false,
    dispatchDrone: false,
    notifyAuthority: false
  }
};

export const scenarioPresets: IncidentState[] = [
  {
    ...base,
    incidentId: "INC-COOKING-001",
    scenarioName: "False alarm: cooking smoke",
    smokePpm: 72,
    temperatureC: 28,
    cameraSmokeConfidence: 0.35,
    cameraFireConfidence: 0.05,
    occupancyStatus: "detected",
    sensorHealth: 0.95,
    historicalFalseAlarmRate: 0.45,
    windSpeedKmh: 12,
    imageUrl: "/sample-images/cooking-smoke.svg"
  },
  {
    ...base,
    incidentId: "INC-FIRE-002",
    scenarioName: "Confirmed fire",
    smokePpm: 115,
    temperatureC: 58,
    cameraSmokeConfidence: 0.88,
    cameraFireConfidence: 0.74,
    occupancyStatus: "unknown",
    sensorHealth: 0.91,
    historicalFalseAlarmRate: 0.08,
    windSpeedKmh: 16,
    imageUrl: "/sample-images/fire-smoke-room.svg"
  },
  {
    ...base,
    incidentId: "INC-LOWCAM-003",
    scenarioName: "Low confidence camera",
    smokePpm: 84,
    temperatureC: 39,
    cameraSmokeConfidence: 0.42,
    cameraFireConfidence: 0.18,
    occupancyStatus: "unknown",
    sensorHealth: 0.87,
    historicalFalseAlarmRate: 0.22,
    windSpeedKmh: 18,
    imageUrl: "/sample-images/unclear-camera.svg"
  },
  {
    ...base,
    incidentId: "INC-SENSOR-004",
    scenarioName: "Sensor degraded",
    smokePpm: 79,
    temperatureC: 35,
    cameraSmokeConfidence: 0.63,
    cameraFireConfidence: 0.21,
    occupancyStatus: "none",
    sensorHealth: 0.48,
    historicalFalseAlarmRate: 0.35,
    windSpeedKmh: 10
  },
  {
    ...base,
    incidentId: "INC-NODRONE-005",
    scenarioName: "Drone unavailable",
    smokePpm: 103,
    temperatureC: 49,
    cameraSmokeConfidence: 0.81,
    cameraFireConfidence: 0.62,
    occupancyStatus: "unknown",
    droneAvailable: false,
    sensorHealth: 0.9,
    historicalFalseAlarmRate: 0.14,
    windSpeedKmh: 20
  }
];

export function clonePreset(index = 0): IncidentState {
  return JSON.parse(JSON.stringify(scenarioPresets[index]));
}

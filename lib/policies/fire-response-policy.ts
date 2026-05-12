export const fireResponsePolicy = {
  version: "fire-response-policy.v1",
  rules: [
    {
      id: "POL-DRONE-ALLOW",
      description:
        "Dispatch drone allowed when drone is available, weather is safe, and fire probability or camera evidence is sufficient."
    },
    {
      id: "POL-DRONE-APPROVAL",
      description:
        "Dispatch drone requires approval under higher wind, weak camera confidence, or unknown occupancy with high risk."
    },
    {
      id: "POL-GATE-APPROVAL",
      description: "Unlock gate always requires human approval and is blocked below high risk."
    },
    {
      id: "POL-AUTHORITY",
      description: "Authority notification is allowed for critical risk and approval-gated for high risk."
    },
    {
      id: "POL-RECHECK",
      description: "Camera recheck is allowed when confidence is low or visual uncertainty exists."
    },
    {
      id: "POL-ALARM-NON-SUPPRESSION",
      description: "The agent can never suppress an alarm."
    }
  ]
};

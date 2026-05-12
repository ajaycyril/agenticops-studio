package agenticops.fire_response

default allow := false
default requires_approval := false

allow if {
  input.action == "notify_operator"
  input.risk != "low"
}

allow if {
  input.action == "dispatch_drone"
  input.droneAvailable == true
  input.windSpeedKmh <= 35
  input.fireProbability >= 0.65
}

requires_approval if {
  input.action == "dispatch_drone"
  input.windSpeedKmh > 25
}

requires_approval if {
  input.action == "unlock_gate"
}

allow if {
  input.action == "unlock_gate"
  input.gateLocked == true
  input.risk == "high"
}

allow if {
  input.action == "unlock_gate"
  input.gateLocked == true
  input.risk == "critical"
}

allow if {
  input.action == "notify_authority"
  input.risk == "critical"
}

package agenticops.drone_dispatch

default allow := false

allow if {
  input.droneAvailable == true
  input.windSpeedKmh <= 35
  input.fireProbability >= 0.65
}

allow if {
  input.droneAvailable == true
  input.windSpeedKmh <= 35
  input.cameraSmokeConfidence >= 0.7
}

allow if {
  input.droneAvailable == true
  input.windSpeedKmh <= 35
  input.cameraFireConfidence >= 0.55
}

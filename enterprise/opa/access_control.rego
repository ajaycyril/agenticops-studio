package agenticops.access_control

default allow_unlock := false
default requires_human_approval := true

allow_unlock if {
  input.risk == "high"
  input.gateLocked == true
}

allow_unlock if {
  input.risk == "critical"
  input.gateLocked == true
}

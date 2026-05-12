package agenticops.authority_notification

default allow := false
default requires_approval := false

allow if {
  input.risk == "critical"
}

requires_approval if {
  input.risk == "high"
}

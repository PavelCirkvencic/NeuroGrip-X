# Health and safety-relevant diagnostics from a read-only AI node.
#
# This message grants no actuator authority. A controller may use validated
# uncertainty inside its predeclared hard limits; invalid data triggers the
# nominal-model fallback and conservative speed envelope.
std_msgs/Header header

bool healthy
string state                 # ok | stale | degraded | out_of_distribution | error
string detail
float32 uncertainty_radius   # conformal radius used for the reported coverage
float32 ood_score            # context Mahalanobis distance
float32 deadline_miss_rate   # fraction of inference deadlines missed
float32 latency_ms

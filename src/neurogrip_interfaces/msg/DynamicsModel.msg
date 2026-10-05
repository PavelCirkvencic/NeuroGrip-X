# Deployable context-conditioned physics-residual dynamics model (physical units).
#
# Prediction: x_next = A_dyn * x + B_dyn * delta, with x = [v_y, yaw_rate].
# A and B are discrete-time physical matrices at sample_time_s (row-major).
# Injected low-level simulator parameters are never part of this message.
std_msgs/Header header

uint32 schema_version                 # 2=GRU, 3=Koopman, 4=grip-aware Koopman
string artifact_sha256
float64 sample_time_s                 # e.g. 0.02
uint32 state_dim                      # 2
uint32 input_dim                      # 1
string[] state_names                  # [v_y_mps, yaw_rate_rps]
string[] input_names                  # [steering_angle_rad]
float64[] a_matrix                    # state_dim * state_dim, row-major
float64[] b_matrix                    # state_dim * input_dim, row-major
float64[] context
float32 ensemble_std
float32 conformal_radius
float32 ood_score
float32 latency_ms
float32 estimated_front_grip         # causal estimate; schema 4 only
float32 estimated_rear_grip          # causal estimate; schema 4 only
float32 front_grip_std               # ensemble disagreement; schema 4 only
float32 rear_grip_std                # ensemble disagreement; schema 4 only
float32 front_grip_error_q90         # held-out absolute-error quantile
float32 rear_grip_error_q90          # held-out absolute-error quantile
bool valid
string invalid_reason

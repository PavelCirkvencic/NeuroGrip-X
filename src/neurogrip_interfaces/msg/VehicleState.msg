# Body-frame vehicle state estimate used by NeuroGrip-X v2 nodes.
#
# Frame: header.frame_id is "base_footprint" (ISO 8855 / SAE style: x forward,
# y left, z up, yaw positive CCW around z). Units are SI.
# source: 0 = EUFS ground truth, 1 = fused/noisy estimate.
std_msgs/Header header

float64 v_x_mps
float64 v_y_mps
float64 yaw_rate_rps
float64 yaw_rad
float64 a_x_mps2
float64 a_y_mps2
float64 steering_applied_rad          # from the patched wheel-speed channel
float64 acceleration_command_mps2
uint8 source

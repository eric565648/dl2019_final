#!/usr/bin/env python
import math
import time
import numpy as np


class LaneControl(object):
    """docstring for LaneControl."""
    def __init__(self):
        super(LaneControl, self).__init__()
        self.arg = 0

        # Driving PID parameters
        self.cross_track_err = 0
        self.heading_err = 0
        self.cross_track_integral = 0
        self.heading_integral = 0
        self.cross_track_integral_top_cutoff = 0.3
        self.cross_track_integral_bottom_cutoff = -0.3
        self.heading_integral_top_cutoff = 1.2
        self.heading_integral_bottom_cutoff = -1.2

        self.actuator_limits_v = 1

        self.v_bar = 0.23
        self.k_d = -3.5
        self.k_theta = -3
        self.d_thres = 0.2615
        self.d_ref = 0
        self.phi_ref = 0
        self.theta_thres = 0.523
        self.d_offset = 0
        self.k_Id = 1
        self.k_Iphi = 0

        self.min_radius = 0.00001

        self.velocity_to_m_per_s = 1
        self.omega_to_rad_per_s = 1

        self.omega_max = 1.5
        self.omega_min = -1.5

        # lane pose configuration
        self.omega_ff = 0

        self.init = True

    def cbPose(self, pose_msg, time_diff):

        action = np.array([0., 0.])

        # self.lane_reading = pose_msg

        # Calculating the delay image processing took
        # timestamp_now = rospy.Time.now()
        # image_delay_stamp = timestamp_now - self.lane_reading.header.stamp
        #
        # # delay from taking the image until now in seconds
        # image_delay = image_delay_stamp.secs + image_delay_stamp.nsecs / 1e9

        print("Pose Msg", pose_msg)
        print("Time diff", time_diff)
        pose_d = -1*pose_msg[0]
        pose_phi = -1*pose_msg[1]

        prev_cross_track_err = self.cross_track_err
        prev_heading_err = self.heading_err

        self.cross_track_err = pose_d- self.d_offset
        self.heading_err = pose_phi

        # car_control_msg = Twist2DStamped()
        # car_control_msg.header = pose_msg.header

        # car_control_msg.v = pose_msg.v_ref
        action[0] = 0.22

        if action[0] > self.actuator_limits_v:
            action[0] = actuator_limits_v

        if math.fabs(self.cross_track_err) > self.d_thres:
            print("inside threshold ")
            self.cross_track_err = self.cross_track_err / math.fabs(self.cross_track_err) * self.d_thres

        # currentMillis = int(round(time.time() * 1000))

        if not self.init:
            # dt = (currentMillis - self.last_ms) / 1000.0
            dt = time_diff
            self.cross_track_integral += self.cross_track_err * dt
            self.heading_integral += self.heading_err * dt

        if self.cross_track_integral > self.cross_track_integral_top_cutoff:
            self.cross_track_integral = self.cross_track_integral_top_cutoff
        if self.cross_track_integral < self.cross_track_integral_bottom_cutoff:
            self.cross_track_integral = self.cross_track_integral_bottom_cutoff

        if self.heading_integral > self.heading_integral_top_cutoff:
            self.heading_integral = self.heading_integral_top_cutoff
        if self.heading_integral < self.heading_integral_bottom_cutoff:
            self.heading_integral = self.heading_integral_bottom_cutoff

        if abs(self.cross_track_err) <= 0.011:  # TODO: replace '<= 0.011' by '< delta_d' (but delta_d might need to be sent by the lane_filter_node.py or even lane_filter.py)
            self.cross_track_integral = 0
        if abs(self.heading_err) <= 0.051:  # TODO: replace '<= 0.051' by '< delta_phi' (but delta_phi might need to be sent by the lane_filter_node.py or even lane_filter.py)
            self.heading_integral = 0
        if np.sign(self.cross_track_err) != np.sign(prev_cross_track_err):  # sign of error changed => error passed zero
            self.cross_track_integral = 0
        if np.sign(self.heading_err) != np.sign(prev_heading_err):  # sign of error changed => error passed zero
            self.heading_integral = 0
        # if self.wheels_cmd_executed.vel_right == 0 and self.wheels_cmd_executed.vel_left == 0:  # if actual velocity sent to the motors is zero
        #     self.cross_track_integral = 0
        #     self.heading_integral = 0

        # omega_feedforward = car_control_msg.v * pose_msg.curvature_ref
        # if self.main_pose_source == "lane_filter" and not self.use_feedforward_part:
        omega_feedforward = 0 # We don't know the situation in front

        # Scale the parameters linear such that their real value is at 0.22m/s TODO do this nice that  * (0.22/self.v_bar)
        omega = self.k_d * (0.22/self.v_bar) * self.cross_track_err + self.k_theta * (0.22/self.v_bar) * self.heading_err
        print("Omega after p:", omega)
        omega += (omega_feedforward)




        # check if nominal omega satisfies min radius, otherwise constrain it to minimal radius
        if math.fabs(omega) > action[0] / self.min_radius:
            if not self.init:
                self.cross_track_integral -= self.cross_track_err * dt
                self.heading_integral -= self.heading_err * dt
            omega = math.copysign(action[0] / self.min_radius, omega)

        # if not self.fsm_state == "SAFE_JOYSTICK_CONTROL":
        #     # apply integral correction (these should not affect radius, hence checked afterwards)
        omega -= self.k_Id * (0.22/self.v_bar) * self.cross_track_integral
        omega -= self.k_Iphi * (0.22/self.v_bar) * self.heading_integral

        print("Omega after I:", omega)

        if action[0] == 0:
            omega = 0
        else:
        # check if velocity is large enough such that car can actually execute desired omega
            if action[0] - 0.5 * math.fabs(omega) * 0.1 < 0.065:
                action[0] = 0.065 + 0.5 * math.fabs(omega) * 0.1




        # apply magic conversion factors
        action[0] = action[0] * self.velocity_to_m_per_s
        action[1] = omega * self.omega_to_rad_per_s

        omega = action[1]
        if omega > self.omega_max: omega = self.omega_max
        if omega < self.omega_min: omega = self.omega_min
        omega += self.omega_ff
        action[1] = omega

        self.init = False

        print("V, Omega: ", action)
        return action

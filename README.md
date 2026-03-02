# Real-World Outdoor Navigation Testing Experiment

This project utilizes the AgileX SCOUT robot as the control chassis. Measuring 69.9 cm wide, 93 cm long, and 34.8 cm high, it achieves a maximum travel speed of 6 km/h. The chassis provides a ROS interface capable of directly outputting linear and angular velocities for robot control. It incorporates a 256-core NVIDIA Pascal™ architecture GPU, 8GB of memory, a 128GB M.2 solid-state drive, and various standard hardware interfaces.

For global positioning to obtain orientation data, the robot incorporates a UBLOX M8030-72CH Global Positioning System (GPS) and a WTGAHRS1 Inertial Measurement Unit (IMU). The camera system utilizes three Raspberry Pi Camera V2 RGB cameras,mounted 0.6 meters above the geometric center of the robot chassis at a downward angle of approximately 12 degrees.

The MobileNetV4 network serves as the skeleton for the boundary line model.
Boundary line information from the three cameras is processed through an attention network.
MapNet employs the A* algorithm to generate a global planning map, which is then cropped to produce a robot-centered local planning map.
The decision model is based on the PPO algorithm.

1. Source the ROS2 setup files with: `source /opt/ros/kilted/setup.bash`.
2. If anything doesn't work, try `ros2 daemon start`.
3. Run Car node via: `ros2 run robotic_car car`
4. To update the uv environment run the following from within the top-level `robotic_car` directory: `uv sync --project dev/`
5. If VSCode is not selecting the correct interpreter ensure it is pointing at: `~/ros2_ws/src/my_package/dev/.venv/bin/python`.
6. When adding a new Node, remember to add it to the `setup.py` file.
@echo off

set PYTHONPATH=%WEBOTS_HOME%\lib\controller\python

::set WEBOTS_CONTROLLER_URL=tcp://192.168.10.18:1234/Erebus_Bot

set WEBOTS_CONTROLLER_URL=tcp://localhost:1234/Erebus_Bot

::python whileRobot2.0.py

python nofocus1.py

pause
# when the wx is "dead" (as seen by wxinfo), or a system reboot has happened, this script is needed.
sudo wxcontrol -r wxpfw0
sudo systemctl restart wolverine
sudo /opt/micron/sbin/wxspt wxcp0 enable
sudo /opt/micron/sbin/wxspt wxcp0 32

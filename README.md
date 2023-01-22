OpenWrt Sysupgrade
==================

Manage [OpenWRT](https://www.openwrt.org/) upgrades easily and with minimal downtime on x86 (and possibly other) hardware targets.

This role installs a script that makes upgrading OpenWrt easy.
It works by installing the new version to a new partition, so that two (or more) versions of OpenWrt are installed at the same time.
It also copies configuration files to the new installation and installs custom packages.
Just reboot into the new version to complete the upgrade.

If the new version is not working correctly, simply boot into the previous installation again.

Note that the upgrade script makes a few assumptions that can usually be met on x86, but not necessarily on other targets:
* Several megabytes need to be available on the root partition to install Python and a few other packages.
* There must be enough unpartitioned space available for at least one more partition of (roughly) the same size as the root partition.

Requirements
------------

This role has no special requirements on the controller.

It does, however, require a working [Python](https://www.python.org/) installation on the target system or [gekmihesg's Ansible library for OpenWRT](https://github.com/gekmihesg/ansible-openwrt) on the Ansible controller.

Role Variables
--------------

* `openwrt_sysupgrade_check_frequency`  
  If set, defines when to check for new OpenWrt releases.
  Must be set to a time specification as understood by cron (cf. `ctrontab(5)`).
  Optional.

Dependencies
------------

This role does not depend on any specific roles.

License
-------

MIT

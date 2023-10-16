#!/usr/bin/env python3
"""OpenWrt sysupgrade helper.

This script handles OpenWrt upgrades by installing the new version to a new partition, thus
enabling multiple parallel installations on the same system.
Configuration files and installed packages from the old installation are copied to the new
installation, so that it only takes a reboot to switch to the new installation.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import logging.handlers
import os.path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import typing as t
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

JSON: t.TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

OPENWRT_DOWNLOAD_SERVER = "https://downloads.openwrt.org/"

_temporary_mounts: dict[Path, tempfile.TemporaryDirectory[str]] = {}


JsonLsblk = t.TypedDict(
    "JsonLsblk",
    {
        "name": str,
        "kname": str,
        "path": str,
        "maj:min": str,
        "fsavail": str | None,
        "fssize": str | None,
        "fstype": str | None,
        "fsused": str | None,
        "fsuse%": str | None,
        "fsroots": list[str | None],
        "fsver": str | None,
        "mountpoint": str | None,
        "mountpoints": list[str | None],
        "label": str | None,
        "uuid": str | None,
        "ptuuid": str,
        "pttype": str,
        "parttype": str | None,
        "parttypename": str | None,
        "partlabel": str | None,
        "partuuid": str | None,
        "partflags": str | None,
        "ra": int,
        "ro": bool,
        "rm": bool,
        "hotplug": bool,
        "model": str,
        "serial": str | None,
        "size": str,
        "state": str,
        "owner": str,
        "group": str,
        "mode": str,
        "alignment": int,
        "min-io": int,
        "opt-io": int,
        "phy-sec": int,
        "log-sec": int,
        "rota": bool,
        "sched": str,
        "rq-size": int,
        "type": str,
        "disk-aln": int,
        "disk-gran": str,
        "disk-max": str,
        "disk-zero": bool,
        "wsame": str,
        "wwn": str | None,
        "rand": bool,
        "pkname": str | None,
        "hctl": str,
        "tran": str | None,
        "subsystems": str,
        "rev": str,
        "vendor": str,
        "zoned": str,
        "dax": bool,
    },
    total=False,
)
JsonPartedDisk = t.TypedDict(
    "JsonPartedDisk",
    {
        "path": str,
        "size": int,
        "transport-type": str,
        "logical-sector-size": int,
        "physical-sector-size": int,
        "partition-table-type": str,
        "model-name": str,
    },
    total=True,
)
JsonPartedPart = t.TypedDict(
    "JsonPartedPart",
    {
        "number": int,
        "begin": int,
        "end": int,
        "size": int,
        "filesystem-type": str | None,
        "partition-name": "t.NotRequired[str | None]",
        "flags-set": "t.NotRequired[str | None]",
    },
    total=True,
)


class PartitionError(Exception):
    """Exception for all errors related to partitions."""


class FilesystemError(Exception):
    """Exception for all errors related to file systems."""


class DownloadError(Exception):
    """Exception for all errors related to downloading files."""


class VersionError(Exception):
    """Exception for all errors related to installed versions of OpenWrt."""


class ArchiveError(Exception):
    """Exception for all errors related to archives."""


class OpkgError(Exception):
    """Exception for all errors related to failed opkg operations.

    Args:
        action: The opkg action that failed.
        root_dir: Root directory that opkg operated on.
        returncode: Exit status of opkg.
        cmd: opkg command that caused the exception.
        stdout: Stdout output of opkg.
        stderr: Stderr output of opkg.
    """

    def __init__(
        self, action: str, root_dir: Path, returncode: int, cmd: str, stdout: str, stderr: str
    ):
        self.action = action
        self.root_dir = root_dir
        self.returncode = returncode
        self.cmd = cmd
        self.stdout = stdout
        self.stderr = stderr

    def __str__(self) -> str:
        return f"Could not {self.action} in {self.root_dir}:\n{self.stderr}"


class Version:
    """Software version object.

    This class represents software versions with exactly three components separated by `.` and
    optionally a release candidate number, appended to the main version after `-rc`.
    This works for OpenWrt version numbers.

    Args:
        version: The version number as a string representation.
    """

    def __init__(self, version: str) -> None:
        self._version = version

    def __lt__(self, other: Version) -> bool:
        """Check if this version is smaller/older than another version.

        Args:
            other: Version to compare to.

        Returns:
            True if this version is smaller/older and false if it is higher/newer or equal.
        """
        if (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch):
            if self.rc == 0 and other.rc > 0:
                return False
            if self.rc > 0 and other.rc == 0:
                return True
            return self.rc < other.rc
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    def __eq__(self, other: object) -> bool:
        return str(self) == str(other)

    def __str__(self) -> str:
        return self._version

    @property
    def major(self) -> int:
        """Return the first component of the version.

        Returns:
            The first component of the version.
        """
        return int(self._version.split(".")[0])

    @property
    def minor(self) -> int:
        """Return the second component of the version.

        Returns:
            The second component of the version.
        """
        return int(self._version.split(".")[1])

    @property
    def patch(self) -> int:
        """Return the third component of the version.

        Returns:
            The third component of the version.
        """
        return int(self._version.split(".")[2].split("-")[0])

    @property
    def rc(self) -> int:  # pylint: disable=invalid-name # 'rc' really is the only sensible choice.
        """Return the release candidate component of the version, if present.

        Returns:
            The release candidate component of the version, if present. If the version does not
            have a release candidate component, return 0.
        """
        try:
            return int(self._version.split("-")[1][2:])
        except IndexError:
            return 0


class AnsiFormatter(logging.Formatter):
    """Formatter for the logging module with limited support for ANSI escape sequences.

    This formatter colours log messages with ANSI codes according to the message's log level.
    Messages can only be colourized in full, not partially.
    Only a few ANSI sequences are supported, namely bold and basic foreground colours.
    The escape sequences are hard-coded and do not take the terminal's capabilities into account.
    """

    reset = 0
    bold = 1

    fg_black = 30
    fg_red = 31
    fg_green = 32
    fg_yellow = 33
    fg_blue = 34
    fg_magenta = 35
    fg_cyan = 36
    fg_white = 37

    colour_map = {
        logging.DEBUG: fg_blue,
        logging.INFO: fg_green,
        logging.WARNING: fg_yellow,
        logging.ERROR: fg_red,
    }
    """Mapping between log levels and their colour codes."""

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record with ANSI escape sequences to colourise the output.

        Args:
            record: The log record to format.

        Returns:
            The log record as a string according to the configured format with added ANSI escape
            sequences.
        """
        return (
            self.esc(self.colour_map[record.levelno])
            + super().format(record)
            + self.esc(self.reset)
        )

    @classmethod
    def esc(cls, code: int | t.Sequence[int]) -> str:
        """Return the escaped ANSI code.

        Args:
            code: An ANSI code or list of ANSI codes to escape.

        Returns:
            The ANSI code(s) with the correct escape sequences around them.
        """
        if isinstance(code, int):
            code = [code]
        return f"\033[{';'.join([str(c) for c in code])}m"


log = logging.getLogger(__name__).log
debug = logging.getLogger(__name__).debug
info = logging.getLogger(__name__).info
warning = logging.getLogger(__name__).warning
error = logging.getLogger(__name__).error


def bytes_to_human(number: int, decimal_places: int = 1) -> str:
    """Return a human-readable representation of the given number of bytes.

    Args:
        number: A number of bytes.
        decimal_places: The number of decimal places to include in the output.

    Returns:
        The human-readable representation of the number of bytes and the unit.
    """
    for unit in ["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB", "RiB", "QiB"]:
        if abs(number) < 1024.0 or unit == "QiB":
            break
        number /= 1024.0  # type: ignore[assignment] # changing to a float is intended
    return f"{number:.{decimal_places}f} {unit}"


def human_to_bytes(size: str) -> int:
    """Parse a human-readable representation of bytes and return it as an integer.

    This function only supports IEC units, not SI units. The 'iB' suffix of the unit may be
    omitted, e.g. 'M' is interpreted as 'MiB'. Numbers without a unit are interpreted as bytes.
    There may or may not be whitespace between the number and the unit. The number may be an
    integer or a float, e.g. '1.5 GiB'.

    Args:
        size: The human-readable representation of bytes.

    Returns:
        The number of bytes that `size` represents.

    Raises:
        ValueError: `size` can not be parsed.
    """
    units = {
        "B": 1,
        "K": 2**10,
        "M": 2**20,
        "G": 2**30,
        "T": 2**40,
        "P": 2**50,
        "E": 2**60,
        "Z": 2**70,
        "Y": 2**80,
        "R": 2**90,
        "Q": 2**100,
    }
    match = re.fullmatch(
        r"(?P<number>\d+(?:\.\d+)?)\s*(?P<unit>[KMGTPEZYRQ])?(?:iB)?", size, re.IGNORECASE
    )
    if match is None:
        raise ValueError(f"Invalid size: {size}")
    number = float(match.group("number"))
    unit = match.group("unit") or "B"
    return int(number * units[unit])


@t.overload
def run(cmd: t.Sequence[str | Path], text: t.Literal[True] = True, **kwargs: t.Any) -> str:
    ...


@t.overload
def run(cmd: t.Sequence[str | Path], text: t.Literal[False], **kwargs: t.Any) -> bytes:
    ...


def run(cmd: t.Sequence[str | Path], text: bool = True, **kwargs: t.Any) -> str | bytes:
    """Run the given command and return its output.

    Args:
        cmd: The command to run as a list of arguments.

    For the other parameters, see `subprocess.run`.

    Returns:
        The output of `cmd` on stdout. This is usually returned as a string, unless `text=False` is
        given in `kwargs`.

    Raises:
        subprocess.SubprocessError: Raised by `subprocess.run`. In particular, when the command
            exited with a non-zero exit code (and `check` is true).
    """
    if not kwargs:
        kwargs = {}
    if "check" not in kwargs:
        kwargs["check"] = True
    if "capture_output" not in kwargs:
        kwargs["capture_output"] = True
    kwargs["text"] = text
    debug(f"Running command: {' '.join([str(c) for c in cmd])}")
    # pylint: disable=subprocess-run-check # check is in kwargs
    result = subprocess.run(cmd, **kwargs).stdout
    if kwargs["text"]:
        return t.cast(str, result)
    return t.cast(bytes, result)


def install(
    packages: str | t.Sequence[str], root_dir: Path = Path("/"), no_action: bool = False
) -> None:
    """Install the given package(s) using `opkg`.

    Args:
        packages: A package name or list of package names to install.
        root_dir: The root directory where the packages should be installed.
        no_action: Do not install packages but test only. Passes the --noaction flag to opkg.
            Still raises an exception on error.

    Raises:
        OpkgError: opkg returned an error.
    """
    if isinstance(packages, str):
        packages = [packages]
    opkg_opts: list[str | Path] = []
    if root_dir != Path("/"):
        opkg_opts += ["--offline-root", root_dir, "--conf", root_dir / "etc" / "opkg.conf"]
    # Update the package lists, if it is older than 30 minutes.
    with open(root_dir / "etc" / "opkg.conf", "r", encoding="utf-8") as opkg_conf:
        for line in opkg_conf:
            if line.startswith("lists_dir ext"):
                lists_dir = root_dir / Path(line[14:].strip().lstrip("/"))
                break
        else:
            lists_dir = root_dir / "var" / "opkg-lists"
    try:
        mtime = datetime.fromtimestamp(lists_dir.stat().st_mtime)
    except FileNotFoundError:
        mtime = datetime.fromtimestamp(0)
    if (
        not lists_dir.exists()
        or not any(lists_dir.iterdir())
        or (datetime.now() - mtime) > timedelta(minutes=30)
    ):
        debug(f"Updating opkg package lists in {root_dir}.")
        try:
            run(["opkg", *opkg_opts, "update"])
        except subprocess.CalledProcessError as ex:
            raise OpkgError(
                action="update package lists",
                root_dir=root_dir,
                returncode=ex.returncode,
                cmd=ex.cmd,
                stdout=ex.stdout,
                stderr=ex.stderr,
            ) from ex
    if no_action:
        opkg_opts.append("--noaction")
    else:
        debug(f"Installing the following package(s) in {root_dir}: {', '.join(packages)}")
    # Install the packages.
    try:
        run(["opkg", *opkg_opts, "install", *packages])
    except subprocess.CalledProcessError as ex:
        raise OpkgError(
            action="install package(s)",
            root_dir=root_dir,
            returncode=ex.returncode,
            cmd=ex.cmd,
            stdout=ex.stdout,
            stderr=ex.stderr,
        ) from ex


def ubus_invoke(
    path: str, method: str, message: t.Optional[t.Mapping[str, "JSON"]] = None
) -> dict[str, "JSON"]:
    """Call a ubus method.

    Args:
        path: The path to the ubus method to call, e.g. `system` or `network.interface.lan`. Run
            `ubus list` for valid paths.
        method: The name of the ubus method to call, e.g. `board`. Run `ubus -v list` for valid
            methods for each path.
        message: The parameters to the called method as a JSON object.

    Returns:
        The JSON formatted data returned by the ubus call.

    Raises:
        OSError: ubus returned an error.
    """
    cmd = ["ubus", "call", path, method]
    if message is not None:
        cmd.append(json.dumps(message))
    try:
        result = run(cmd)
    except subprocess.CalledProcessError as ex:
        raise OSError(f"Ubus call {path}.{method} failed:\n{ex.stderr}") from ex
    return json.loads(result)


def lsblk(columns: t.Iterable[str] | None = None, device: Path | None = None) -> list[JsonLsblk]:
    """Run `lsblk` and return the output.

    Args:
        columns: A list of columns to include in the output. If not set, a default set of columns
            is returned.
        device: A device path to run `lsblk` on. If not set, all devices are queried.

    Returns:
        The JSON formatted data returned by `lsblk`.

    Raises:
        PartitionError: lsblk returned an error.
    """
    cmd: list[str | Path] = ["lsblk", "--json", "--list"]
    if columns is not None:
        cmd += ["-o", ",".join(columns)]
    if device is not None:
        cmd += [device]
    try:
        result = run(cmd)
    except subprocess.CalledProcessError as ex:
        raise PartitionError(f"lsblk failed:\n{ex.stderr}") from ex
    return json.loads(result)["blockdevices"]


def parted(
    disk: Path, commands: t.Sequence[str | Path]
) -> tuple[JsonPartedDisk | JsonPartedPart, ...]:
    # Return type should be `tuple[JsonPartedDisk, t.Unpack[tuple[JsonPartedPart, ...]]]`,
    # but mypy 0.991 can not yet handle that.
    """Run `parted` and return the output.

    Args:
        disk: Path that references the disk's device node, e.g. `/dev/sda`.
        commands: A list containing one or more `parted` commands and their parameters.

    Returns:
        The output of `parted` parsed into a JSON structure.

    Raises:
        PartitionError: parted returned an error.
    """
    cmd: list[str | Path] = ["parted", "--script", "--machine", disk, "unit", "B", *commands]
    try:
        parted_output = run(cmd)
    except subprocess.CalledProcessError as ex:
        raise PartitionError(f"parted failed:\n{ex.stderr}") from ex
    lines = [line.rstrip(";") for line in parted_output.splitlines()]
    if not lines:
        return ()
    # Parse disk information (second line).
    result: list[JsonPartedDisk | JsonPartedPart] = [
        {
            "path": lines[1].split(":")[0],
            "size": int(lines[1].split(":")[1].rstrip("B")),
            "transport-type": lines[1].split(":")[2],
            "logical-sector-size": int(lines[1].split(":")[3]),
            "physical-sector-size": int(lines[1].split(":")[4]),
            "partition-table-type": lines[1].split(":")[5],
            "model-name": lines[1].split(":")[6],
        }
    ]
    # Parse partition information (third line and following).
    for partition in lines[2:]:
        partinfo: JsonPartedPart = {
            "number": int(partition.split(":")[0]),
            "begin": int(partition.split(":")[1].rstrip("B")),
            "end": int(partition.split(":")[2].rstrip("B")),
            "size": int(partition.split(":")[3].rstrip("B")),
            "filesystem-type": partition.split(":")[4] or None,
        }
        try:
            partinfo["partition-name"] = partition.split(":")[5] or None
        except IndexError:
            pass
        try:
            partinfo["flags-set"] = partition.split(":")[6] or None
        except IndexError:
            pass
        result.append(partinfo)
    return tuple(result)


@contextlib.contextmanager
def mount(partition: Path) -> t.Generator[Path, None, None]:
    """Mount the given partition to a temporary path and return that path.

    Temporary paths are registered and if the partition is already mounted, the temporary path is
    returned without mounting it again.
    This function can be used as a context manager that automatically unmounts the partition.

    Args:
        partition: Path that references the partition's device node, e.g. `/dev/sda2`.

    Yields:
        The path where the partition is mounted.

    Raises:
        OSError: The partition could not be mounted.
    """
    if partition in _temporary_mounts:
        yield Path(_temporary_mounts[partition].name)
    # Create a temporary mount point.
    tmpdir = tempfile.TemporaryDirectory(prefix="sysupgrade-")
    # Mount the partition.
    debug(f"Mounting {partition} to {tmpdir.name}.")
    try:
        run(["mount", "-o", "noatime", partition, tmpdir.name])
    except subprocess.CalledProcessError as ex:
        raise OSError(f"Could not mount {partition}:\n{ex.stderr}") from ex
    # Store the TemporaryDirectory object.
    _temporary_mounts[partition] = tmpdir
    try:
        # Return the path.
        yield Path(_temporary_mounts[partition].name)
    finally:
        # Clean up.
        unmount(partition)


def unmount(partition: Path) -> None:
    """Unmount the given partition if it is mounted to a temporary path.

    Args:
        partition: Path that references the partition's device node, e.g. `/dev/sda2`.

    Raises:
        OSError: The partition could not be unmounted.
    """
    if partition in _temporary_mounts:
        # Unmount the partition.
        debug(f"Unmounting {partition} from {_temporary_mounts[partition].name}.")
        try:
            run(["umount", _temporary_mounts[partition].name])
        except subprocess.CalledProcessError as ex:
            raise OSError(f"Could not unmount {partition}:\n{ex.stderr}") from ex
        # Remove the path. This also deletes the directory from the file system.
        del _temporary_mounts[partition]


def extract_tarfile(file: Path, destination: Path) -> None:
    """Securely extract all files from a (possibly compressed) tar archive.

    The documentation of the `tarfile` module advises to not extract files from untrusted archives
    without prior inspection. This function is an attempt at such an inspection.
    It checks the archive members for absolute paths, path traversal attempts and symlink attacks.

    Args:
        file: Path to a (possibly compressed) tar archive.
        destination: Path to a directory to which the archive is to be extracted.

    Raises:
        ArchiveError: The archive contains files that would be extracted outside the destination
            directory.
    """
    debug(f"Extracting tar archive {file} to {destination}.")
    with tarfile.open(file, "r:*") as tar:
        symlinks: list[str] = []
        for member in tar.getmembers():
            try:
                # Check for archive members that would extract outside the target directory.
                # This includes path traversal attempts and absolute paths.
                (destination / member.name).resolve().relative_to(destination)
            except ValueError as ex:
                raise ArchiveError(f"Attempted path traversal in {file}: {member.name}") from ex
            for link in symlinks:
                try:
                    # Check for archive members that are relative to some symlink in the archive.
                    # They might extract outside the target directory.
                    # The following statement raises ValueError if the file is *not* relative to
                    # the symlink, which is fine.
                    Path(os.path.normpath(member.name)).relative_to(Path(link))
                    raise ArchiveError(f"Attempted path traversal in {file}: {member.name}")
                except ValueError:
                    pass
            if member.issym():
                # The archive member is a symlink. Note its name to check later archive members
                # against it.
                symlinks.append(member.name)
        # Extract all the files to the new root file system.
        tar.extractall(destination, numeric_owner=True)


def get_current_version() -> Version:
    """Get and return the version of OpenWrt currently running on this system.

    Returns:
        The version number of OpenWrt that is currently running on this system.
    """
    debug("Getting currently running OpenWrt version.")
    return Version(t.cast(dict[str, str], ubus_invoke("system", "board")["release"])["version"])


def get_target() -> str:
    """Get and return the hardware target of the OpenWrt installation.

    Returns:
        The hardware target name of the OpenWrt installation on this system.
    """
    debug("Getting currently running OpenWrt hardware target.")
    return t.cast(dict[str, str], ubus_invoke("system", "board")["release"])["target"]


def get_latest_version() -> Version:
    """Get and return the latest OpenWrt release version.

    Returns:
        The version number of the latest OpenWrt release.

    Raises:
        DownloadError: Could not contact the OpenWrt download server or could not parse the answer.
    """
    debug(f"Querying {OPENWRT_DOWNLOAD_SERVER} for the latest OpenWrt version.")
    try:
        with urllib.request.urlopen(f"{OPENWRT_DOWNLOAD_SERVER}/releases/") as response:
            charset = response.headers.get_content_charset() or "utf-8"
            versions = re.findall(
                r"(?<=>)[0-9]{2}\.[0-9]{2}\.[0-9]+(?=</a>)", response.read().decode(charset)
            )
    except urllib.request.HTTPError as ex:
        raise DownloadError(
            f"Could not get latest OpenWrt release from {OPENWRT_DOWNLOAD_SERVER}/releases/"
        ) from ex
    if len(versions) == 0:
        raise DownloadError("Could not parse available OpenWrt releases.")
    versions.sort(key=Version)
    return Version(versions[-1])


def get_kernel_partition() -> Path:
    """Get and return the device of the partition containing the kernel and boot loader.

    The partition is detected by the file system's label, which needs to be `kernel`. This is the
    case for x86 OpenWrt images, at least. If this does not return exactly one result, the
    partition is detected by the `boot` or `esp` partition flags. If this does not return exactly
    one result either, an exception is raised.

    Returns:
        The path of the device node of the partition that contains the OpenWrt kernel and boot
        loader, e.g. `/dev/sda1`.

    Raises:
        PartitionError: The kernel device could not be uniquely identified.
    """
    debug("Getting kernel and boot loader partition.")
    partitions = lsblk(["PATH", "LABEL", "PARTFLAGS", "PARTTYPE"])
    # Get the partitions that are labeled as `kernel`.
    kernel_partitions = [p for p in partitions if p["label"] == "kernel"]
    if len(kernel_partitions) == 1:
        result = Path(kernel_partitions[0]["path"])
        debug(f"Kernel and boot loader partition is {result}.")
        return result
    warning("No partition is labeled 'kernel'. Partition detection may fail.")
    # Fall back to checking the 'boot' and 'esp' partition flags.
    kernel_partitions = [
        p for p in partitions if p["partflags"] == "0x80" or p["parttype"] == "0xef"
    ]
    if len(kernel_partitions) == 0:
        raise PartitionError("Could not find kernel and boot loader partition.")
    if len(kernel_partitions) > 1:
        raise PartitionError("Found multiple kernel partitions: {', '.join(kernel_partitions)}")
    result = Path(kernel_partitions[0]["path"])
    debug(f"Kernel and boot loader partition is {result}.")
    return result


def get_disk_format(disk: Path) -> str:
    """Get and return the format of the partition table on the given disk.

    Args:
        disk: Path that references the disk's device node, e.g. `/dev/sda`.

    Returns:
        Name of the partition table format, e.g. `dos` or `gpt`.
    """
    partitions = lsblk(["PATH", "PTTYPE"], disk)
    return [p for p in partitions if p["path"] == str(disk)][0]["pttype"]


def get_root_partition(version: Version | None = None) -> Path:
    """Get and return the path of the root partition of the given or current OpenWrt version.

    The version numbers are obtained from the file system labels of the root partitions. Hence,
    the `version` parameter only works, if the root file systems are labeled accordingly.

    Args:
        version: The OpenWrt version for which to return the root partition. Defaults to the
            currently running installation.

    Returns:
        The path of the device node of the root partition of the given OpenWrt version. If no
        version is given, return the partition that is currently mounted on `/`, e.g. `/dev/sda2`.

    Raises:
        PartitionError: The root partition could not be uniquely identified.
    """
    partitions = lsblk(["PATH", "MOUNTPOINTS", "LABEL"])
    if version is None:
        debug("Getting root partition for current installation.")
        root_partitions = [p for p in partitions if "/" in p["mountpoints"]]
    else:
        debug(f"Getting root partition for OpenWrt version {version}.")
        root_partitions = [p for p in partitions if p["label"] == f"rootfs-{version}"]
    if len(root_partitions) == 0:
        raise PartitionError(
            "Could not find root partition for "
            f"{'current installation' if version is None else 'OpenWrt ' + str(version)}."
        )
    if len(root_partitions) > 1:
        raise PartitionError(
            "Found multiple root partitions for "
            f"{'current installation' if version is None else 'OpenWrt ' + str(version)}: "
            "{', '.join(root_partitions)}"
        )
    result = Path(root_partitions[0]["path"])
    debug(f"Root partition is {result}.")
    return result


def get_installed_versions() -> list[Version]:
    """Get and return all installed versions of OpenWrt on this system.

    The version numbers are obtained from the file system labels of the root partitions.

    Returns:
        A list of all installed OpenWrt versions, sorted from oldest to newest version number.
    """
    # Get the labels of all partitions, filter those that look like root partitions, extract their
    # version number and convert it to a Version object. In a single list comprehension.
    debug("Getting all currently installed OpenWrt versions.")
    versions = [
        Version(p["label"][7:])
        for p in lsblk(["LABEL"])
        if p["label"] is not None and p["label"].startswith("rootfs-")
    ]
    versions.sort()
    return versions


def get_disk(partition: Path) -> Path:
    """Get and return the disk on which the given partition resides.

    Args:
        partition: Path that references the partition's device node, e.g. `/dev/sda2`.

    Returns:
        Path of the device node of the disk containing `partition`.
    """
    pkname = lsblk(["PKNAME"], partition)[0]["pkname"]
    assert pkname is not None
    return Path("/dev/" + pkname)


def get_partition_number(partition: Path) -> int:
    """Return the number of the partition, e.g. `2` for `/dev/sda2`.

    Args:
        partition: Path that references the partition's device node, e.g. `/dev/sda2`.

    Returns:
        The partition's number.
    """
    match = re.fullmatch(
        f"{re.escape(str(get_disk(partition)))}p?(?P<partnum>[0-9]+)", str(partition)
    )
    assert match is not None
    return int(match.group("partnum"))


def get_partuuid(partition: Path) -> str:
    """Get and return the partuuid of the given partition.

    Args:
        partition: Path that references the partition's device node, e.g. `/dev/sda2`.

    Returns:
        The partuuid of the given partition.
    """
    result = lsblk(["PARTUUID"], partition)
    assert len(result) == 1
    assert result[0]["partuuid"] is not None
    return result[0]["partuuid"]


def get_filesystem(partition: Path) -> str | None:
    """Return the name of the file system on the given partition.

    Args:
        partition: Path that references the partition's device node, e.g. `/dev/sda2`.

    Returns:
        Name of the file system on the partition or `None`, if no file system can be detected.
    """
    return lsblk(["FSTYPE"], partition)[0]["fstype"]


def get_partition_size(partition: Path) -> int:
    """Get and return the size of the given partition.

    Args:
        partition: Path that references the partition's device node, e.g. `/dev/sda2`.

    Returns:
        The size of `partition` in bytes.
    """
    partition_number = get_partition_number(partition)
    partitions = parted(get_disk(partition), ["print"])[1:]
    return [p for p in partitions if p["number"] == partition_number][0]["size"]  # type: ignore[typeddict-item]


def get_oldest_installation() -> Version | None:
    """Get and return the OpenWrt version number of the oldest installation on this system.

    Returns:
        The smallest version number of all OpenWrt installations on this system or `None`, if the
        current installation is the only installation.
    """
    debug("Getting oldest installed OpenWrt version.")
    # Get the installed versions and filter out the version of the running installation.
    versions = [v for v in get_installed_versions() if v != get_current_version()]
    if versions:
        debug(f"Oldest installed OpenWrt version is {versions[0]}.")
        return versions[0]
    debug("No old version of OpenWrt detected.")
    return None


def count_primary_partitions(disk: Path) -> int:
    """Return the number of primary/extended partitions on the given disk.

    Args:
        disk: Path that references the disk's device node, e.g. `/dev/sda`.

    Returns:
        The number of primary and extended partitions on the disk, if the disk uses a DOS partition
        table. Otherwise, 0 is returned.

    Raises:
        PartitionError: parted returned an error.
    """
    if get_disk_format(disk) != "dos":
        # The disk does not have a DOS partition table and, therefore, does not have primary
        # partitions.
        return 0
    # Note: We can not use the `parted` function here, since the machine-readable output does not
    # include the partition type.
    try:
        output = run(["parted", "--script", disk, "print"])
    except subprocess.CalledProcessError as ex:
        raise PartitionError(f"parted failed:\n{ex.stderr}") from ex
    return output.count("primary") + output.count("extended")


def get_new_partition_start(disk: Path, size: int) -> int | None:
    """Get and return the position where a partition of the given size can be created on the disk.

    Args:
        disk: Path that references the disk's device node, e.g. `/dev/sda`.
        size: The size requirement for the partition in bytes.

    Returns:
        The minimal position where the partition can be created in bytes.
        If there is no sufficiently large unallocated block on the disk, `None` is returned.
    """
    debug(f"Trying to find {bytes_to_human(size)} of free space on {disk}.")
    # Note: Each block is a JsonPartedPart (not a JsonPartedDisk) and therefore guaranteed to
    # have the keys mypy complains about. This is why "typeddict-item" errors are ignored here.
    for block in [
        p for p in parted(disk, ["print", "free"])[1:] if p["filesystem-type"] == "free"  # type: ignore[typeddict-item]
    ]:
        if block["size"] >= size:
            debug(
                f"Found free space block on {disk} at position {bytes_to_human(block['begin'])}."  # type: ignore[typeddict-item]
            )
            return block["begin"]  # type: ignore[typeddict-item]
    debug(f"Did not find a large enough free space block on {disk}.")
    return None


def set_label(partition: Path, label: str) -> None:
    """Set the file system and partition label.

    Partition labels (or names) are only supported on GPT (and a few other exotic partition tables)
    and silently ignored when not supported. In this case, only the file system label is set.

    Args:
        partition: Path that references the partition's device node, e.g. `/dev/sda2`.
        label: The file system and partition label to set.

    Raises:
        FilesystemError: The file system on the given partition is unknown or not supported.
    """
    info(f"Setting label {label} on {partition}.")
    filesystem = get_filesystem(partition)
    if filesystem is None:
        raise FilesystemError(f"Unknown file system on {partition}.")
    if filesystem.startswith("ext"):
        run(["tune2fs", "-L", label, partition])
    else:
        raise FilesystemError(f"Unsupported file system {filesystem} on {partition}.")
    disk = get_disk(partition)
    if get_disk_format(disk) in ["gpt", "mac", "pc98"]:
        partition_number = get_partition_number(partition)
        parted(disk, ["name", str(partition_number), label])


def download_openwrt(version: Version, target_dir: Path) -> tuple[Path, Path]:
    """Download the kernel and root file system archive for the given OpenWrt version.

    This function verifies the integrity of the downloaded files using the cryptographic signature
    of the sha256sums file. The public key with which to verify the signature needs to be in
    `/etc/opkg/keys/`.

    Args:
        version: The version of OpenWrt which should be downloaded.
        target_dir: The (temporary) directory where the downloaded files should be stored.

    Returns:
        The paths of the downloaded kernel and root file system.

    Raises:
        DownloadError: The target version of OpenWrt could not be downloaded or the downloaded
            files failed verification.
    """
    target = get_target()
    base_url = f"https://downloads.openwrt.org/releases/{version}/targets/{target}"
    kernel_name = f"openwrt-{version}-{target.replace('/', '-')}-generic-kernel.bin"
    rootfs_name = f"openwrt-{version}-{target.replace('/', '-')}-rootfs.tar.gz"
    info(f"Downloading OpenWrt {version}.")

    # Download the kernel and rootfs and their checksums.
    for file in [kernel_name, rootfs_name, "sha256sums", "sha256sums.sig"]:
        debug(f"Downloading {file} from {base_url}/{file}.")
        file_path = target_dir / file
        try:
            with (
                urllib.request.urlopen(f"{base_url}/{file}") as response,
                open(file_path, "wb") as f,
            ):
                shutil.copyfileobj(response, f)
        except urllib.request.HTTPError as ex:
            raise DownloadError(f"Could not download {file}: {ex}") from ex

    # Check the integrity of the downloaded files.
    # Check the signature on sha256sums (using sha256sums.sig).
    try:
        debug("Verifying cryptographic signature on sha256sums.")
        run(["usign", "-q", "-V", "-m", target_dir / "sha256sums", "-P", "/etc/opkg/keys"])
    except subprocess.CalledProcessError as ex:
        raise DownloadError("Checksums file has an invalid signature.") from ex
    with open(target_dir / "sha256sums", "r", encoding="utf-8") as f:
        sha256sums = f.readlines()
    # Check that the kernel's SHA-256 hash matches.
    debug("Verifying kernel image checksum.")
    with open(target_dir / kernel_name, "rb") as f:
        kernel_hash = hashlib.sha256(f.read()).hexdigest()
    if f"{kernel_hash} *{kernel_name}\n" not in sha256sums:
        raise DownloadError("Kernel image has an invalid checksum.")
    # Check that the rootfs's SHA-256 hash matches.
    debug("Verifying rootfs archive checksum.")
    with open(target_dir / rootfs_name, "rb") as f:
        rootfs_hash = hashlib.sha256(f.read()).hexdigest()
    if f"{rootfs_hash} *{rootfs_name}\n" not in sha256sums:
        raise DownloadError("Rootfs archive has an invalid checksum.")

    return (target_dir / kernel_name, target_dir / rootfs_name)


def delete_installation(version: Version) -> None:
    """Delete the installation of the given OpenWrt version.

    This deletes the partition, the kernel image and the entries in grub.cfg.

    Args:
        version: The OpenWrt version that is to be deleted.

    Raises:
        VersionError: Attempt to remove the currently running version.
    """
    # Do not delete the currently running version.
    if version == get_current_version():
        raise VersionError(f"Can not remove currently running OpenWrt version {version}.")

    info(f"Deleting OpenWrt installation for version {version}.")
    with mount(get_kernel_partition()) as tmpdir:
        versioned_kernel = tmpdir / "boot" / f"vmlinuz-{version}"

        # Delete the grub.cfg entries.
        with open(tmpdir / "boot" / "grub" / "grub.cfg", "r", encoding="utf-8") as f:
            grub_cfg = f.read()
        modified_grub_cfg = re.sub(
            rf"""^\s*menuentry "OpenWrt.*{{
\s*linux\s+/boot/{re.escape(versioned_kernel.name)}\s.*
\s*}}$""",
            "",
            grub_cfg,
            flags=re.MULTILINE,
        )
        if modified_grub_cfg != grub_cfg:
            debug(f"Removing grub.cfg entries for {version}.")
            with open(tmpdir / "boot" / "grub" / "grub.cfg", "w", encoding="utf-8") as f:
                f.write(modified_grub_cfg)

        # Delete the kernel image.
        if versioned_kernel.exists():
            debug(f"Deleting kernel image for {version}.")
            versioned_kernel.unlink()

    # Delete the partition(s). It should never be more than one, really.
    for partition in [p for p in lsblk(["PATH", "LABEL"]) if p["label"] == f"rootfs-{version}"]:
        match = re.search("(?P<partnum>[0-9]+)$", partition["path"])
        assert match is not None
        partnum = match.group("partnum")
        debug(f"Deleting OpenWrt {version} rootfs partition {partition}.")
        parted(get_disk(Path(partition["path"])), ["rm", partnum])


def create_rootfs_partition(version: Version, size: int | None = None) -> Path:
    """Create a new partition for an OpenWrt root file system and return its path.

    Args:
        version: Number of the OpenWrt version which will be installed on this partition. This
            becomes part of the partition label on gpt-partitioned disks but is otherwise ignored.
        size: Size of the new partition in bytes. If not set, the size of the current root file
            system is copied.

    Returns:
        The path of the new partition's device node, e.g. `/dev/sda3`.

    Raises:
        PartitionError: A new rootfs partition could not be created.
    """
    # Get some information.
    root_partition = get_root_partition()
    root_disk = get_disk(root_partition)
    root_disk_format = get_disk_format(root_disk)
    if size is None:
        size = get_partition_size(root_partition)

    # Delete old installations to free up space for a new installation, if necessary.
    if root_disk_format == "dos" and count_primary_partitions(root_disk) >= 4:
        # DOS partition tables are limited to four primary/extended partitions.
        # The limit as been reached, so we need to delete an installation to be able to create
        # a new primary partition.
        oldest_version = get_oldest_installation()
        debug(
            f"{root_disk} can not hold another primary partition. "
            f"Removing OpenWrt {oldest_version} installation to free space."
        )
        if oldest_version is None:
            raise PartitionError(
                f"Can not create primary partition on {root_disk} and no old installation to delete."
            )
        delete_installation(oldest_version)
        assert count_primary_partitions(root_disk) < 4
    for _ in range(3):
        # Try up to three times to free up space, if necessary.
        free_block = get_new_partition_start(root_disk, size)
        if free_block is None:
            oldest_version = get_oldest_installation()
            if oldest_version is None:
                raise PartitionError(
                    f"Not enough unpartitioned space on {root_disk} and no old installation to delete."
                )
            debug(
                f"{root_disk} does not have enough free space for another rootfs partition."
                f"Removing OpenWrt {oldest_version} installation to free space."
            )
            delete_installation(oldest_version)
        else:
            break
    else:
        raise PartitionError(f"Could not free a block of {bytes_to_human(size)} on {root_disk}.")

    root_filesystem = get_filesystem(root_partition) or "ext4"

    # Create new rootfs partition.
    info(f"Creating new rootfs partition for OpenWrt {version} on {root_disk}.")
    old_partitions = [p["path"] for p in lsblk(["PATH"], root_disk)]
    if root_disk_format == "dos":
        parted(
            root_disk,
            ["mkpart", "primary", root_filesystem, str(free_block), str(free_block + size)],
        )
    elif root_disk_format == "gpt":
        parted(
            root_disk,
            [
                "mkpart",
                f"rootfs-{version}",
                root_filesystem,
                str(free_block),
                str(free_block + size),
            ],
        )
    else:
        raise PartitionError("Unsupported partition table format.")
    new_partitions = [p["path"] for p in lsblk(["PATH"], root_disk)]
    return Path([path for path in new_partitions if path not in old_partitions][0])


def copy_config_files(target_dir: Path) -> None:
    """Copy config files from the current installation of OpenWrt to the target directory.

    Args:
        target_dir: Copy the config files to this directory. Should be a root file system mount
            point.

    Raises:
        OSError: Creating the configuration file backup archive failed.
        ArchiveError: Extracting the configuration file backup archive failed.
    """
    info("Copying configuration files from current installation.")
    with tempfile.TemporaryDirectory(prefix="sysupgrade-") as tmpdir:
        try:
            run(["sysupgrade", "-b", Path(tmpdir) / "backup.tar.gz"])
        except subprocess.CalledProcessError as ex:
            raise OSError("Could not create backup of configuration files.") from ex
        extract_tarfile(Path(tmpdir) / "backup.tar.gz", target_dir)


def copy_packages(target_dir: Path) -> None:
    """Install currently installed packages on the installation in the target directory.

    This function installs all currently installed packages that were manually installed
    (i.e. not as a dependency of another package) and are not already installed in the target
    directory.

    Args:
        target_dir: Target directory for package installation. Needs to contain an OpenWrt
            installation.

    Raises:
        OSError: Mounting or unmounting /tmp in the target rootfs failed.
        OpkgError: Installing packages to the target rootfs failed.
    """
    info("Copying installed packages from current installation.")
    # Get all installed packages.
    installed_packages = [
        line.split(" - ")[0] for line in run(["opkg", "list-installed"]).splitlines()
    ]
    # Mount /tmp on the new root partition as opkg requires it.
    try:
        run(["mount", "-o", "nosuid,nodev,noatime", "-t", "tmpfs", "tmpfs", target_dir / "tmp"])
    except subprocess.CalledProcessError as ex:
        raise OSError(f"Could not mount tmpfs on {target_dir / 'tmp'}:\n{ex.stderr}") from ex
    try:
        # opkg also requires /var/lock (which is symlinked to /tmp).
        (target_dir / "var/lock").mkdir(mode=1777)
        custom_packages: list[str] = []
        for pkg in installed_packages:
            # Skip packages that are not explicitly installed, but merely to satisfy a dependency,
            # i.e. those that opkg does not call "user installed".
            status = run(["opkg", "status", pkg])
            if "user installed" not in status:
                continue
            # Skip packages that are not marked as "user installed", but have other installed
            # packages depending on them.
            reverse_deps = run(["opkg", "whatdepends", pkg])
            if not reverse_deps.splitlines()[-1].startswith("What depends on"):
                continue
            # Skip packages that are already installed on the new partition (because they
            # are part of the default installation).
            status = run(
                [
                    "opkg",
                    "--offline-root",
                    target_dir,
                    "--conf",
                    target_dir / "etc" / "opkg.conf",
                    "status",
                    pkg,
                ]
            )
            if "install" in status:
                continue
            # Not skipped, note it for installation.
            custom_packages.append(pkg)
        # Install the "user installed" packages on the new partition.
        install(custom_packages, target_dir)
    finally:
        try:
            run(["umount", target_dir / "tmp"])
        except subprocess.CalledProcessError as ex:
            raise OSError(f"Could not unmount tmpfs on {target_dir / 'tmp'}:\n{ex.stderr}") from ex


def add_to_grub(version: Version) -> None:
    """Add an entry in the grub menu for the given OpenWrt version.

    The entries are made by copying and adjusting the entries for the currently running version of
    OpenWrt.

    Args:
        version: The version of OpenWrt for which a grub menu entry is to be added. Must be
            installed on this system.

    Raises:
        VersionError: Unable to find menu entries for the current version.
    """
    info(f"Adding menu entries for OpenWrt {version} to grub.cfg.")
    current_version = get_current_version()
    with mount(get_kernel_partition()) as boot_dir:
        # Read current grub.cfg.
        with open(boot_dir / "boot" / "grub" / "grub.cfg", "r", encoding="utf-8") as f:
            grub_cfg = f.read()
        # Extract entries for the current installation.
        current_version_entries = re.findall(
            rf"""^\s*menuentry "OpenWrt.*{{
\s*linux\s+/boot/vmlinuz-{re.escape(str(current_version))}\s.*
\s*}}$""",
            grub_cfg,
            flags=re.MULTILINE,
        )
        if len(current_version_entries) == 0:
            raise VersionError(
                "Could not find boot entries in grub.cfg for the current OpenWrt installation."
            )
        # Modify the entries for the current installation to reference the new installation.
        new_version_entries: list[str] = []
        for entry in current_version_entries:
            entry = entry.strip().replace(str(current_version), str(version))
            entry = re.sub(
                r"(?<=root=)\S+(?=\s)",
                f"PARTUUID={get_partuuid(get_root_partition(version))}",
                entry,
            )
            new_version_entries.append(entry)
        # Get the first "menuentry" line in grub.cfg.
        grub_cfg_lines = grub_cfg.splitlines()
        assert len(grub_cfg_lines) > 0
        i = 0
        for i, line in enumerate(grub_cfg_lines):
            if line.strip().startswith("menuentry"):
                break
        # Insert the new entries just before the first "menuentry" line.
        grub_cfg = "\n".join(grub_cfg_lines[:i] + new_version_entries + grub_cfg_lines[i:]) + "\n"
        # Write the new grub.cfg.
        with open(boot_dir / "boot" / "grub" / "grub.cfg", "w", encoding="utf-8") as f:
            f.write(grub_cfg)


def init(args: argparse.Namespace) -> int:
    # pylint: disable=unused-argument # 'args' is required by the subcommands' common interface.
    """Prepare the system for managing OpenWrt upgrades using this script.

    Args:
        args: Command line arguments as parsed by argparse.

    Returns:
        0

    Raises:
        FilesystemError: The file system on the root partition is unknown or not supported.
    """
    info("Preparing system for OpenWrt upgrade management.")
    # Install dependencies.
    install(["lsblk", "parted"])
    # Get some information about the partitions and filesystems.
    kernel_partition = get_kernel_partition()
    root_partition = get_root_partition()
    root_filesystem = get_filesystem(root_partition)
    if root_filesystem is None:
        raise FilesystemError(f"Unknown root file system on {root_partition}.")
    # Install filesystem tools.
    if root_filesystem.startswith("ext"):
        install(["e2fsprogs", "tune2fs"])
    else:
        raise FilesystemError(f"Unsupported root file system {root_filesystem}.")
    current_version = get_current_version()
    # Change the filesystem label of the root partition to include the OpenWrt version.
    set_label(root_partition, f"rootfs-{current_version}")
    # Mount the kernel partition.
    with mount(kernel_partition) as tmpdir:
        # Rename the kernel image to include the OpenWrt version.
        kernel = tmpdir / "boot" / "vmlinuz"
        versioned_kernel = tmpdir / "boot" / f"vmlinuz-{current_version}"
        if kernel.exists():
            debug(f"Renaming kernel image to {versioned_kernel.name}.")
            kernel.rename(versioned_kernel)
        assert versioned_kernel.exists()
        # Change grub.cfg to reference the renamed kernel image.
        with open(tmpdir / "boot" / "grub" / "grub.cfg", "r", encoding="utf-8") as f:
            grub_cfg = f.read()
        modified_grub_cfg = grub_cfg
        modified_grub_cfg = re.sub(
            rf"/boot/{re.escape(kernel.name)}(?=\s)",
            f"/boot/{versioned_kernel.name}",
            modified_grub_cfg,
        )
        modified_grub_cfg = re.sub(
            r'(?<=menuentry\s"(?i:OpenWrt))(?=( \(failsafe\))?")',
            f" {current_version}",
            modified_grub_cfg,
        )
        if modified_grub_cfg != grub_cfg:
            debug("Changing menu entries in grub.cfg.")
            with open(tmpdir / "boot" / "grub" / "grub.cfg", "w", encoding="utf-8") as f:
                f.write(modified_grub_cfg)
    return 0


def list_versions(args: argparse.Namespace) -> int:
    # pylint: disable=unused-argument # 'args' is required by the subcommands' common interface.
    """List installed versions of OpenWrt.

    Args:
        args: Command line arguments as parsed by argparse.

    Returns:
        0
    """
    for version in get_installed_versions():
        print(version)
    return 0


def check(args: argparse.Namespace) -> int:
    # pylint: disable=unused-argument # 'args' is required by the subcommands' common interface.
    """Check if a newer release of OpenWrt is available.

    Args:
        args: Command line arguments as parsed by argparse.

    Returns:
        1 if an upgrade is available and 0 otherwise.
    """
    info("Checking for a new OpenWrt release.")
    latest_version = get_latest_version()
    if get_current_version() < latest_version:
        print(f"An upgrade to OpenWrt {latest_version} is available. To upgrade, run:")
        print(f"{__file__} upgrade")
        return 1
    print("No upgrade available.")
    return 0


def upgrade(args: argparse.Namespace) -> int:
    """Upgrade the system to the given version of OpenWrt.

    Args:
        args: Command line arguments as parsed by argparse.

    Returns:
        0

    Raises:
        FilesystemError: The root file system of the current installation is unknown or not
            supported by this script or the file system on the new partition could not be created.
        VersionError: The given version of OpenWrt is already installed.
    """
    root_partition = get_root_partition()
    root_partition_size = get_partition_size(root_partition)
    root_filesystem = get_filesystem(root_partition)
    if root_filesystem is None:
        raise FilesystemError(f"Unknown root file system on {root_partition}.")

    if args.version is None:
        args.version = get_latest_version()
    if args.version in get_installed_versions():
        raise VersionError(f"OpenWrt {args.version} is already installed.")

    info(f"Upgrading OpenWrt installation to {args.version}.")

    if args.size is not None:
        if args.size.startswith("+"):
            debug(f"New rootfs partition size will be increased by {args.size[1:]}.")
            root_partition_size += human_to_bytes(args.size[1:])
        elif args.size.startswith("-"):
            debug(f"New rootfs partition size will be reduced by {args.size[1:]}.")
            root_partition_size -= human_to_bytes(args.size[1:])
        else:
            debug(f"New rootfs partition size will be changed to {args.size}.")
            root_partition_size = human_to_bytes(args.size)

    with tempfile.TemporaryDirectory(prefix="sysupgrade-") as tmpdir_str:
        tmpdir = Path(tmpdir_str)

        # Download the target version of OpenWrt.
        kernel_path, rootfs_path = download_openwrt(args.version, tmpdir)

        # Create a new rootfs partition.
        new_root_partition = create_rootfs_partition(args.version, root_partition_size)

        # Install the kernel.
        debug(f"Installing OpenWrt {args.version} kernel image to kernel partition.")
        with mount(get_kernel_partition()) as boot_dir:
            shutil.move(kernel_path, boot_dir / "boot" / f"vmlinuz-{args.version}")

        # Format the new partition.
        if root_filesystem.startswith("ext"):
            debug(f"Creating {root_filesystem} on {new_root_partition}.")
            try:
                run(
                    [
                        f"mkfs.{root_filesystem}",
                        "-F",
                        "-L",
                        f"rootfs-{args.version}",
                        new_root_partition,
                    ]
                )
            except subprocess.CalledProcessError as ex:
                raise FilesystemError(
                    f"Could not create {root_filesystem} on {new_root_partition}:\n{ex.stderr}"
                ) from ex
        else:
            raise FilesystemError(f"Unsupported root filesystem {root_filesystem}.")

        with mount(new_root_partition) as root_dir:

            # Install the rootfs files to the new partition.
            debug(f"Installing OpenWrt {args.version} to {new_root_partition}.")
            extract_tarfile(rootfs_path, root_dir)

            # Copy configuration files from the current installation.
            copy_config_files(root_dir)

            # Install custom packages from the current installation.
            copy_packages(root_dir)

    # Add the new installation to grub.cfg.
    add_to_grub(args.version)

    print(f"OpenWrt {args.version} was installed successfully.")
    print("Please reboot the system to finish the upgrade.")

    return 0


def remove(args: argparse.Namespace) -> int:
    """Remove one or more old installations of OpenWrt.

    Args:
        args: Command line arguments as parsed by argparse.

    Returns:
        0

    Raises:
        VersionError: Attempt to remove the currently running version.
    """
    current_version = get_current_version()
    for version in args.version:
        if version == current_version:
            raise VersionError(
                f"Can not remove running version {version}. Please boot into another installation "
                "and try again."
            )
        delete_installation(version)
    return 0


def prune(args: argparse.Namespace) -> int:
    """Remove the oldest installation of OpenWrt.

    Args:
        args: Command line arguments as parsed by argparse.

    Returns:
        0
    """
    current_version = get_current_version()
    old_versions = [v for v in get_installed_versions() if v != current_version]
    if args.keep > 0:
        old_versions = old_versions[: -args.keep]
    for version in old_versions:
        delete_installation(version)
    return 0


def main() -> t.NoReturn:
    """Main entry point into the script."""
    parser = argparse.ArgumentParser(
        description="Manage parallel installation of multiple OpenWrt versions."
    )
    parser.add_argument(
        "--verbose",
        "-v",
        dest="log_level",
        action="append_const",
        const=-1,
        default=[2],
        help="Print and log more status messages. Adding multiple -v will increase the verbosity.",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        dest="log_level",
        action="append_const",
        const=1,
        help="Print and log fewer status messages. Adding multiple -q will decrease the verbosity.",
    )
    subs = parser.add_subparsers(dest="action", title="action", required=True)

    parser_init = subs.add_parser(
        "init", help="Prepare the system for managing OpenWrt upgrades using this script."
    )
    parser_init.set_defaults(func=init)

    parser_list = subs.add_parser("list", help="List installed versions of OpenWrt.")
    parser_list.set_defaults(func=list_versions)

    parser_check = subs.add_parser(
        "check", help="Check if a newer release of OpenWrt is available."
    )
    parser_check.set_defaults(func=check)

    parser_upgrade = subs.add_parser("upgrade", help="Upgrade to the given version of OpenWrt.")
    parser_upgrade.add_argument("version", nargs="?", type=Version)
    parser_upgrade.add_argument(
        "--size",
        help="Change the partition size for the new installation to this value. "
        "Prefix with +/- to use the size of the current root partition increased/decrease by "
        "this number.",
    )
    parser_upgrade.set_defaults(func=upgrade)

    parser_remove = subs.add_parser(
        "remove", help="Remove one or more old installations of OpenWrt."
    )
    parser_remove.add_argument("version", nargs="+", type=Version)
    parser_remove.set_defaults(func=remove)

    def _non_negative_int(value: str) -> int:
        result = int(value)
        if result < 0:
            raise argparse.ArgumentTypeError("can not be negative")
        return result

    parser_prune = subs.add_parser("prune", help="Remove the oldest installations of OpenWrt.")
    parser_prune.add_argument(
        "--keep",
        "-k",
        type=_non_negative_int,
        default=1,
        help="Keep this many old versions (default: %(default)s).",
    )
    parser_prune.set_defaults(func=prune)

    args = parser.parse_args()

    # Set up logger to write to console and syslog consistently.
    args.log_level = min(max(sum(args.log_level), 0), 3)
    logger = logging.getLogger(__name__)
    logger.setLevel([logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR][args.log_level])
    console_handler = logging.StreamHandler()
    if sys.stderr.isatty():
        console_handler.setFormatter(AnsiFormatter("%(levelname)s: %(message)s"))
    else:
        console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    syslog_handler = logging.handlers.SysLogHandler(address="/dev/log")
    syslog_handler.setFormatter(logging.Formatter("[%(filename)s] %(message)s"))
    logger.addHandler(console_handler)
    logger.addHandler(syslog_handler)

    try:
        sys.exit(args.func(args))
        # pylint: disable=broad-except # Catch *all* unhandled exceptions from the subcommand is
        # the point here.
    except Exception as ex:
        if args.log_level == 0:
            error(str(ex), exc_info=sys.exc_info())
        else:
            error(str(ex))
        sys.exit(2)


if __name__ == "__main__":
    main()

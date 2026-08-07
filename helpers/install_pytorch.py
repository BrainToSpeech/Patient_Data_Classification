"""Install the PyTorch 2.6 wheel appropriate for this computer."""

import argparse
import platform
import re
import shutil
import subprocess
import sys


TORCH_VERSION = "2.6.0"
COMPUTE_INDEXES = {
    "cpu": "cpu",
    "cu118": "cu118",
    "cu124": "cu124",
    "cu126": "cu126",
}


def detected_compute_platform():
    if platform.system() == "Darwin":
        return "macos"

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return "cpu"

    try:
        output = subprocess.check_output(
            [nvidia_smi], text=True, stderr=subprocess.STDOUT
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "nvidia-smi is installed but unavailable. Fix the NVIDIA driver or "
            "rerun with --compute cpu/cu118/cu124/cu126."
        ) from error

    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", output)
    if match is None:
        raise RuntimeError(
            "Could not read the supported CUDA version from nvidia-smi. "
            "Use --compute to select it explicitly."
        )

    supported_cuda = tuple(map(int, match.groups()))
    if supported_cuda >= (12, 6):
        return "cu126"
    if supported_cuda >= (12, 4):
        return "cu124"
    if supported_cuda >= (11, 8):
        return "cu118"
    raise RuntimeError(
        f"The NVIDIA driver only reports CUDA {supported_cuda[0]}.{supported_cuda[1]}. "
        "Update the driver or rerun with --compute cpu."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compute",
        choices=["auto", "cpu", "cu118", "cu124", "cu126"],
        default="auto",
        help="Override automatic GPU detection.",
    )
    args = parser.parse_args()

    try:
        compute = detected_compute_platform() if args.compute == "auto" else args.compute
    except RuntimeError as error:
        parser.error(str(error))
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        f"torch=={TORCH_VERSION}",
        f"torchaudio=={TORCH_VERSION}",
    ]
    if compute in COMPUTE_INDEXES:
        command.extend(
            [
                "--index-url",
                f"https://download.pytorch.org/whl/{COMPUTE_INDEXES[compute]}",
            ]
        )

    print(f"Installing PyTorch for: {compute}", flush=True)
    subprocess.check_call(command)


if __name__ == "__main__":
    main()

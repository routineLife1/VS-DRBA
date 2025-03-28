import os

import requests
from tqdm import tqdm


def download_model(url: str) -> None:
    filename = url.split("/")[-1]
    r = requests.get(url, stream=True)

    save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, filename), "wb") as f:
        with tqdm(
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            miniters=1,
            desc=filename,
            total=int(r.headers.get("content-length", 0)),
        ) as pbar:
            for chunk in r.iter_content(chunk_size=4096):
                f.write(chunk)
                pbar.update(len(chunk))


if __name__ == "__main__":
    url = "https://github.com/HolyWu/vs-rife/releases/download/model/"
    models = [
        "flownet_v4.0",
        "flownet_v4.1",
        "flownet_v4.2",
        "flownet_v4.3",
        "flownet_v4.4",
        "flownet_v4.5",
        "flownet_v4.6",
        "flownet_v4.7",
        "flownet_v4.8",
        "flownet_v4.9",
        "flownet_v4.10",
        "flownet_v4.11",
        "flownet_v4.12",
        "flownet_v4.12.lite",
        "flownet_v4.13",
        "flownet_v4.13.lite",
        "flownet_v4.14",
        "flownet_v4.14.lite",
        "flownet_v4.15",
        "flownet_v4.15.lite",
        "flownet_v4.16.lite",
        "flownet_v4.17",
        "flownet_v4.17.lite",
        "flownet_v4.18",
        "flownet_v4.19",
        "flownet_v4.20",
        "flownet_v4.21",
        "flownet_v4.22",
        "flownet_v4.22.lite",
        "flownet_v4.23",
        "flownet_v4.24",
        "flownet_v4.25",
        "flownet_v4.25.lite",
        "flownet_v4.25.heavy",
        "flownet_v4.26",
        "flownet_v4.26.heavy",
    ]
    for model in models:
        download_model(url + model + ".pkl")

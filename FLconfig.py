import os


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


os.environ["https_proxy"] = "http://proxy.rd.francetelecom.fr:8080"

os.environ["http_proxy"] = os.environ["https_proxy"]

os.environ[
    "no_proxy"
] = "0.0.0.0,::,127.0.0.1,localhost,10.0.0.0/8,192.168.0.0/16,172.16.0.0/16,*.rd.francetelecom.fr"

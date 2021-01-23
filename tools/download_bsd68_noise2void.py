from torchvision.datasets.utils import download_file_from_google_drive, extract_archive

data_folder = "./datasets/"
id_noise = "1lZ89OKOVGKi-4l2-_Vj3Kn_UtGP-faXS"
download_file_from_google_drive(id_noise, data_folder, "bsd68_gaussian25.npy")
id_clean = "1QtLwqayxT4vVmzQ1tEE6Tbk4fKvwPdPI"
download_file_from_google_drive(id_clean, data_folder, "bsd68_groundtruth.npy")
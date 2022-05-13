import gdown
import os
import configargparse
import zipfile

def config_parser():
    parser = configargparse.ArgumentParser(description="Downloader script")   
    # dataset options
    parser.add_argument("--data_path", type=str, help="Data directory to download the datasets", default="data/data/")
    parser.add_argument("--dataset", type=str, choices=["ava", "all", "reddit", "pccd"], default="pccd", help="Dataset(s) to download.")
    return parser


if __name__ == "__main__":

    parser = config_parser()
    args = parser.parse_args()

    if "pccd" in args.dataset:
        
        # pccd_path = os.path.join(args.data_path, "PCCD")
        # if not os.path.exists(pccd_path):
        #     os.mkdir(pccd_path)
        output_file= os.path.join(args.data_path, 'PCCD.zip')
        
        url = "https://drive.google.com/file/d/1hap2UGI9XV5XmxKOo54wZW30OXbqNyo8/view"
        gdown.download(url=url, output=output_file, quiet=False, fuzzy=True)

        with zipfile.ZipFile(output_file, 'r') as zip_ref:
            zip_ref.extractall(args.data_path)

        os.remove(output_file)


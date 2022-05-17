import gdown
import os
import configargparse
import zipfile
import subprocess
import json 
import shutil

def config_parser():
    parser = configargparse.ArgumentParser(description="Downloader script")   
    # dataset options
    parser.add_argument("--data_path", type=str, help="Data directory to download the datasets", default="data/")
    parser.add_argument("--dataset", type=str, choices=["ava", "all", "reddit", "pccd", "dpc"], default=["dpc",], nargs="+", help="Dataset(s) to download.")
    return parser


if __name__ == "__main__":

    parser = config_parser()
    args = parser.parse_args()

    if "pccd" in args.dataset:
        
        if not os.path.exists(args.data_path):
            os.mkdir(args.data_path)
        output_file= os.path.join(args.data_path, 'PCCD.zip')
        
        url = "https://drive.google.com/file/d/1hap2UGI9XV5XmxKOo54wZW30OXbqNyo8/view"
        gdown.download(url=url, output=output_file, quiet=False, fuzzy=True)

        with zipfile.ZipFile(output_file, 'r') as zip_ref:
            zip_ref.extractall(args.data_path)

        os.remove(output_file)

    if "ava" in args.dataset:
        dataset_path= args.data_path + 'ava/'
        if not os.path.exists(dataset_path):
            os.mkdir(dataset_path)
        output_file= os.path.join(dataset_path, 'ava_comments.txt')
        
        url = "https://drive.google.com/file/d/1yUKZpnyIqmyQMBTswmLRnMPgSIthgzmf/view?usp=sharing"
        gdown.download(url=url, output=output_file, quiet=False, fuzzy=True)
    
    if "dpc" in args.dataset:
        dataset_path = args.data_path + 'dpc/'
        if not os.path.exists(dataset_path):
            os.mkdir(dataset_path)
        subprocess.check_call(['git', 'clone', "https://github.com/BestiVictory/DPC-Captions.git", str(dataset_path)])

        attributes = ['color_lighting', 'composition', 'depth_and_focus', 'impression_and_subject', 'use_of_camera']
        
        datasets = []
        for attribute in attributes:
            file = dataset_path + attribute + '.json'
            with open(file, 'r') as f:
                data = json.load(f)
            
            dataset = []
            for k,v in data.items():
                dataset.append(
                    {'im_id': k,
                    'comments': v, 
                    'attribute': attribute}
                )
            datasets.extend(dataset)

            os.remove(file)
        shutil.rmtree(dataset_path + '.git/')
        os.remove(dataset_path + 'README.md')

        with open(f'{dataset_path}dpc.json', 'w') as f:
            json.dump(datasets, f, indent=1)


        
            

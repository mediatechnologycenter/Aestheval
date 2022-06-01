from aestheval.data.datautils.data_downloader import download_datasets
from aestheval.text_prediction.sentiment_score import *
# from aestheval.baselines.run_baseline import run
import configargparse

def config_parser():
    parser = configargparse.ArgumentParser(description="Downloader script")   
    # dataset options
    parser.add_argument("--data_path", type=str, help="Data directory to download the datasets", default="data/")
    parser.add_argument("--download_data", action='store_true', help="Whether to download data", default=False)
    parser.add_argument("--compute_sentiment_score", action='store_true', help="Whether to compute the sentiment score", default=False)
    return parser

if __name__ == "__main__":

    parser = config_parser()
    args = parser.parse_args()

    if args.download_data:
        download_datasets(data_path=args.data_path, dataset='all')
    
    if args.compute_sentiment_score:
        # sentiment_pccd(root_dir=args.data_path)
        sentiment_reddit(root_dir=args.data_path)
        sentiment_ava(root_dir=args.data_path)
#    run('Reddit', 'nima')
#    run('PCCD', 'nima')
#    run('PCCD', 'mlsp')
    # run('Reddit', 'mlsp')
    # run('AVA', 'mlsp')
    # run('AVA', 'nima')
    # run('PCCD', 'vit')
    # run('Reddit', 'vit')
    # run('AVA', 'vit')
from aestheval.data.datautils.data_downloader import download_datasets
from aestheval.text_prediction.sentiment_score import *
from aestheval.baselines.run_baseline import run
from aestheval.baselines.probbing import run as run_probbing
import configargparse

def config_parser():
    parser = configargparse.ArgumentParser(description="Downloader script")   
    # dataset options
    parser.add_argument("--data_path", type=str, help="Data directory to download the datasets", default="data/")
    parser.add_argument("--download_data", action='store_true', help="Whether to download data", default=False)
    parser.add_argument("--compute_sentiment_score", action='store_true', help="Whether to compute the sentiment score", default=False)
    parser.add_argument("--dataset_name", type=str, choices=['PCCD', 'Reddit', 'AVA'], required=True)
    parser.add_argument("--evaluate", action='store_true', help="Whether to run only evaluation")
    parser.add_argument("--model_name", type=str, required=True)
    return parser

if __name__ == "__main__":
    parser = config_parser()
    args = parser.parse_args()

    if args.download_data:
        download_datasets(data_path=args.data_path, dataset='all')
    
    if args.compute_sentiment_score:
        sentiment_pccd(root_dir=args.data_path)
        sentiment_reddit(root_dir=args.data_path)
        sentiment_ava(root_dir=args.data_path)
    
    # run(args.dataset_name, 'nima', args.data_path, args.evaluate)
    run_probbing(args.dataset_name, args.model_name, args.data_path)

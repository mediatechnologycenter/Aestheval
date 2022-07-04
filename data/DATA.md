# How to get the data

1. Download files from __Zenodo:__ https://zenodo.org/record/6656802#.YqyS6xuxWhA and save them in this directory.
      1. 1. You can save them in any directory and pass it as an argument later. 
2. Download AVA + DPC, PCCD and Reddit datasets (WIP, AVA is not completely downloaded yet. We recommend checking [this repo](https://github.com/imfing/ava_downloader) out):

```
python main.py --download_data
```
You can set an specific dataset to download with the argument `--dataset_name`. 

To download data from Reddit, you will need to register a new user agent to query Reddit's API. You have to provide the `CLIENT_ID`, `CLIENT_SECRET` and `USER_AGENT` as environment variables (in a `.env` file for example). You can find more info about how to get the credentials in [Reddit's API wiki](https://github.com/reddit-archive/reddit/wiki/OAuth2-Quick-Start-Example#first-steps). Have in mind you'll have to comply with [Reddit's API rules](https://github.com/reddit-archive/reddit/wiki/API).  

**To download only the images of RPCD dataset:**
```
python main.py --download_data --dataset_name Reddit --only_images
```
This will use the file  ``reddit_photocritique_posts.pkl``, you can pass the directory where you downloaded it with the argument `--data_dir` (default is `data/`).
Remove the `--only_images` flag to download osts and comments too.

## Disclaimer

Getting the data from Reddit and Pushshift can take some hours, depending mainly on their servers load. Data download is implemented with multithreadig using [PMAW](https://github.com/mattpodolak/pmaw).
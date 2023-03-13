# How to get the data

1. Download files from [__Zenodo:__](https://zenodo.org/record/6985507) and save them under `aestheval/data/reddit` in case the ids files are not already there.

2. Download AVA + DPC, PCCD and Reddit datasets (WIP, AVA is not completely downloaded yet. We recommend checking [this repo](https://github.com/imfing/ava_downloader) out):

```
python main.py --download_data --dataset_name Reddit
```
You can set an specific dataset to download with the argument `--dataset_name`. 

To download data from Reddit, you will need to register a new user agent to query Reddit's API. You have to provide the `CLIENT_ID`, `CLIENT_SECRET` and `USER_AGENT` as environment variables (in a `.env` file for example). You can find more info about how to get the credentials in [Reddit's API wiki](https://github.com/reddit-archive/reddit/wiki/OAuth2-Quick-Start-Example#first-steps). Have in mind you'll have to comply with [Reddit's API rules](https://github.com/reddit-archive/reddit/wiki/API).  

## Disclaimer

Getting the data from Reddit and Pushshift can take some time, depending mainly on their servers load. Data download is implemented with multithreadig using [PMAW](https://github.com/mattpodolak/pmaw). We have also encountered SSL certificate errors coming from PMAW while retrieving the content, but it's possible to re-start the download skipping the already downloaded files.


## Data license

The `LICENSE` file at the root of this repo states the license of the code used in this project. Regarding the actual data (namely, images and text) we comply with:

- [Reddit User Agreement](https://www.redditinc.com/policies/user-agreement/),
- [Reddit API terms of use](https://docs.google.com/a/reddit.com/forms/d/e/1FAIpQLSezNdDNK1-P8mspSbmtC2r86Ee9ZRbC66u929cG2GX0T9UMyw/viewform)
- [PushShift database Creative Commons License](https://zenodo.org/record/3608135\#.Yp3XEXZBw2w)

In particular, we refer to the Section 2.d of Reddit API Terms of Use, which states:
_“User Content. Reddit user photos, text and videos (“User Content”) are owned by the users and not by Reddit. Subject to the terms and conditions of these Terms, Reddit grants You a non-exclusive, non-transferable, non-sublicensable, and revocable license to copy and display the User Content using the Reddit API through your application, website, or service to end users. You may not modify the User Content except to format it for such display. You will comply with any requirements or restrictions imposed on usage of User Content by their respective owners, which may include “all rights reserved” notices, Creative Commons licenses or other terms and conditions that may be agreed upon between you and the owners.”_

**We do not provide access to the images directly, but a URL to the corresponding post and image as we do not have the license of those.** This URL may be used to retrieve the images using the provided tools under other researchers’ personal license to use the Reddit API. Moreover, we do not modify the original content by no means, while we provide the necessary tools to process the data and run the same experiments we carried out. 

We release the dataset associated software under the Creative Commons Attribution 4.0 International license.

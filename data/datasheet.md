# Datasheet

We detail the datasheet for documenting the proposed dataset following the [Datasheet for Datasets](https://arxiv.org/abs/1803.09010) template. Note that, while we do not provide any data other that the IDs associated to Reddit posts, we answer the questionnaire considering the constructed dataset resulted from using our code.

## Motivation

### For what purpose was the dataset created?
RPCD was created to drive the research progress in both image aesthetic assessment and aesthetic image captioning. The proposed dataset addresses the need for images acquired with modern acquisition devices and photo critiques that give a better understanding of how the aesthetic evaluation is carried out.
### Who created this dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)?
This dataset was created by the authors on behalf of their respective institutions, ETH Media Technology Center and University of Milano-Biococca.
### Who funded the creation of the dataset?
The creation of this dataset was carried out as part of the [Aesthetic Assessment of Image and Video Content project](https://mtc.ethz.ch/research/image-video-processing/aesthetics-assessment.html). The project is supported by Ringier, TX Group, NZZ, SRG, VSM, viscom, and the ETH Zurich Foundation on the ETH MTC side.

## Composition
### What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)?
Each instance is represented as a tuple containing one image and several photo critiques, where the images are JPEG files and the photo critiques are in textual form.

### How many instances are there in total (of each type, if appropriate)?
RPCD consists of 73,965 data instances. Specifically, there are 73,965 images and 219,790 photo critiques.

### Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set?
The dataset contains all samples (posts) available at the moment of collection, from the origin of the forum until the moment of collection. Additionally, included posts had to meet the following criteria:
– The post has at least an image which could be retrieved.
– The post has at least one comment critiquing the image
– The post is not a discussion thread, a type of post to encourage general discussion in the forum.

### What data does each instance consist of?
Each data instance consists of an image and one or more textual photo critiques.

### Is there a label or target associated with each instance? If so, please provide a description.

There is no label associated with each sample. However, in this work we propose a method
to compute said label, which is calculated using the processing scripts.


### Is any information missing from individual instances?

Some of the samples in the dataset might be missing at the moment of future retrievals due to the users removing the data from Reddit.

### Are relationships between individual instances made explicit (e.g., users’ movie ratings, social network links)?

Every image and comment in the dataset is associated with the user who created the post. Moreover, we build the tree of comments of the different users criticizing an image. However, the data is downloaded by using only the post IDs.

### Are there recommended data splits (e.g., training, development/validation, testing)?

We provide the data splits we used in our experiments in the repository and they are used to retrieve the posts we used, although we encourage the use of other splits. The splits were randomly generated to divide the dataset in 70% train, 10% validation and 20% test splits.

### Are there any errors, sources of noise, or redundancies in the dataset?

The source of data itself could be considered a source of noise. Additionally, we have not evaluated the case in which an image is posted by an user several times in different posts, although we consider this event to be non-existent or insignificant.

### Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)?

The dataset links to resources available on Reddit and Pushshift. In particular, posts and their metadata (including the URLs to images) are retrieved from Pushshift, while the comments are retrieved directly from Reddit. There is no guarantee that the dataset will remain constant, as this depends on the users exercising their rights to remove their content from the dataset sources. For this same reason, there are not any archival versions of the complete dataset
available online. In order to retrieve the dataset in the future, Reddit API credentials are needed. Please, refer to the [instructions about how to obtain the credentials](https://www.reddit.com/wiki/api/).

### Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor-patient confidentiality, data that includes the content of individuals’ non-public communications)?

The dataset does not contain any confidential data as both images and comments are publicly available in Reddit.

### Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?

There are data samples depicting explicit nudity with aesthetics purposes, and we acknowledge that this may be problematic for some people. According to the subreddit rules, this content must be marked: "Not Suitable for Work (NSFW) must be marked. [...] Please keep NSFW posts respectful. Nothing that would be considered pornography." For this reason, the dataset processing script creates a NSFW column in the dataframe to easily filter this content.

### Does the dataset relate to people?

Yes, some of the images contain people or the main subject is a person.

### Does the dataset identify any subpopulations (e.g., by age, gender)?

No.

### Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset?

All posts and comments are linked to users, which may be identifiable depending on the data made available by the user. Additionally, posts and comments may contain information linking to other social media which could serve to identify a certain user.

### Does the dataset contain data that might be considered sensitive in any way (e.g., data that reveals racial or ethnic origins, sexual orientations, religious beliefs, political opinions or union memberships, or locations; financial or health data; biometric or genetic data; forms of government identification, such as social security numbers; criminal history)?

The retrieved data might contain sensitive data publicly disclosed by the users. However, we do not expect this to be common at all, and we would be surprised that some kinds of sensitive information are present in the community (financial, health, biometric, genetic or governmental data).

## Collection process

### How was the data associated with each instance acquired?

The data was directly observable (posts in Reddit stored in Pushshift’s and Reddit’s servers).

### What mechanisms or procedures were used to collect the data (e.g., hardware apparatus or sensor, manual human curation, software program, software API)?

Software API to access both Reddit and Pushshift.

### If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)? 

NA.

### Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)?

Nobody was involved in the data collection process since all data was already available and observable.

### Over what timeframe was the data collected?

Tha data was collected in February 2022, and comprises posts and comments in the span from May 2009 (first posts in the subreddit) to February 2022 (collection date).

### Were any ethical review processes conducted (e.g., by an institutional review board)?

No ethical review process was conducted previous to the ethical review of this conference.

### Does the dataset relate to people?

Yes, but not exclusively.

### Did you collect the data from the individuals in question directly, or obtain it via third parties or other sources (e.g., websites)?

Third party sources (Reddit and Pushshift).

### Were the individuals in question notified about the data collection?

No.

### Did the individuals in question consent to the collection and use of their data?

According to [Reddit’s Privacy Policy](https://www.reddit.com/policies/privacy-policy), which is accepted by every user upon registration, "Reddit also allows third parties to access public Reddit content via the Reddit API and other similar technologies." . Moreover, we note that no data from the users is made directly available in the dataset. It only contains the IDs of the posts and the tools to retrieve them from Reddit and Pushshift.

### If consent was obtained, were the consenting individuals provided with a mechanism to revoke their consent in the future or for certain uses?

Users may remove their data from Reddit and Pushshift using their respective privacy enforcing mechanisms. Thus, they would be removing their data from the dataset.

### Has an analysis of the potential impact of the dataset and its use on data subjects (e.g., a data protection impact analysis) been conducted?

As we note above, no data from the users is made directly available in the dataset. The dataset only contains the IDs of the posts and the tools to retrieve them from Reddit and Pushshift.

## Processing/cleaning/labeling

### Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)?

We provide scrips to automatically process the downloaded raw posts. Only first level comments are kept, posts with no comments or whose image is no longer available are filtered.

### Was the "raw" data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)?

The raw posts need to be downloaded for further processing.

### Is the software used to preprocess/clean/label the instances available?

Yes. The software for downloading and preparing the dataset is available on our GitHub repository 25.

## Uses

### Has the dataset been used for any tasks already?

RPCD is introduced and used in the paper Understanding Aesthetics with Language: A Photo Critique Dataset for Aesthetic Assessment.

### Is there a repository that links to any or all papers or systems that use the dataset?

Papers using RPCD will be listed on the [PapersWithCode web page](https://paperswithcode.com/dataset/rpcd).

### What (other) tasks could the dataset be used for?

RPCD can be used for modelling works in the areas of knowledge retrieval and multimodal reasoning.

### Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?

No, there are no known risks to the best of our knowledge.

### Are there tasks for which the dataset should not be used?

RPCD should not be used for automatically judging a photographer’s skills based on the photo critiques. The latter, in fact, are to be understood as highly subjective judgments that depend on the emotions and background of the commentators and could go beyond the mere
technical evaluation of the shot.

## Distribution

### Will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created?

Yes, the dataset is made publicly accessible.

### How will the dataset will be distributed (e.g., tarball on website, API, GitHub)?

See our [GitHub repository](https://github.com/mediatechnologycenter/aestheval) for downloading instructions. RPCD has the following DOI: 10.5281/zenodo.6985507.

### When will the dataset be distributed?

RPCD will be released to the public in August 2022.

### Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?

We release the dataset under the [Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/).

### Have any third parties imposed IP-based or other restrictions on the data associated with the instances?

No.

### Do any export controls or other regulatory restrictions apply to the dataset or to individual instances?
No.

## Maintenance

### Who is supporting/hosting/maintaining the dataset?

RPCD is supported and maintained by ETH MTC and University of Milano-Bicocca. The post IDs are available on Zenodo, the posts are on Reddit and Pushshift, and the code for automatically retrieving the posts is on GitHub.

### How can the owner/curator/manager of the dataset be contacted (e.g., email address)?

By emailing to {daniel.veranieto,clabrador}@inf.ethz.ch or luigi.celona@unimib.it. By opening an issue on this repository.

### Is there an erratum?
All changes to the dataset will be announced on our [Zenodo repository](https://doi.org/10.5281/zenodo.6985507) and on this repo.

### Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances’)?

All updates (if necessary) will be posted on our [Zenodo repository](https://doi.org/10.5281/zenodo.6985507).

### If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (e.g., were individuals in question told that their data would be retained for a fixed period of time and then deleted)?

The data related to users is stored on Reddit and Pushshift servers, and their data retention policies apply.

### Will older versions of the dataset continue to be supported/hosted/maintained?

All changes to the dataset will be announced on our Zenodo repository. Outdated versions will be kept around for consistency.

### If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?

Any extension/augmentation by an external party is allowed under the release license. The dataset could be easily extended with other communities and other time periods using the available scripts. In order to add the extended version to the existing repositories, please contact the authors.